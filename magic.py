import asyncio
import json
import os
import threading
import wave
import io
import platform
import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from functools import partial
from typing import Optional, Any

import aiohttp
import pyaudio
import simpleaudio as sa
from dotenv import load_dotenv
from google.cloud import speech, texttospeech
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit, QFrame, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QMovie
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# GPIO 모듈 로드 (라즈베리 파이에서만)
if platform.system() == "Linux":
    try:
        from gpiozero import Button
    except ImportError:
        Button = None
        print("gpiozero 패키지가 설치되지 않았습니다. 'pip install gpiozero'를 실행하세요.")
else:
    Button = None

from logging.handlers import RotatingFileHandler

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# 환경 변수 로드
load_dotenv()

API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
if not API_KEY:
    logger.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

GOOGLE_CREDS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or ""
if not GOOGLE_CREDS:
    logger.error("GOOGLE_APPLICATION_CREDENTIALS 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"


class WorkerSignals(QObject):
    message: pyqtSignal = pyqtSignal(str)
    start_processing_image: pyqtSignal = pyqtSignal()
    start_tts_image: pyqtSignal = pyqtSignal()
    stop_tts_image: pyqtSignal = pyqtSignal()


class TTSManager:
    """Google Cloud Text-to-Speech를 관리하는 클래스."""

    def __init__(self, tts_client: texttospeech.TextToSpeechClient, signals: WorkerSignals) -> None:
        self.tts_client: texttospeech.TextToSpeechClient = tts_client
        self.signals: WorkerSignals = signals
        self.play_obj_lock: threading.Lock = threading.Lock()
        self.current_play_obj: Optional[sa.PlayObject] = None

    async def synthesize_and_play(self, text: str) -> None:
        """텍스트를 TTS로 변환하고 재생."""
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code="ko-KR",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            loop = asyncio.get_event_loop()
            response: texttospeech.SynthesizeSpeechResponse = await loop.run_in_executor(
                None,
                partial(
                    self.tts_client.synthesize_speech,
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config
                )
            )

            audio_content: bytes = response.audio_content
            audio_stream = io.BytesIO(audio_content)
            wave_read = wave.Wave_read(audio_stream)
            wave_obj: sa.WaveObject = sa.WaveObject.from_wave_read(wave_read)

            with self.play_obj_lock:
                self.current_play_obj = wave_obj.play()
            logger.info("Google TTS 음성 재생 시작.")
            await loop.run_in_executor(None, self.current_play_obj.wait_done)
            logger.info("Google TTS 음성 재생 완료.")

        except Exception as e:
            logger.error(f"TTS 재생 중 오류: {e}")
        finally:
            self.signals.stop_tts_image.emit()


class STTManager:
    """Google Cloud Speech-to-Text를 관리하는 클래스."""

    def __init__(self, stt_client: speech.SpeechClient) -> None:
        self.stt_client: speech.SpeechClient = stt_client

    async def transcribe_audio(self, audio_file_path: str) -> str:
        """오디오 파일을 텍스트로 변환."""
        try:
            with open(audio_file_path, 'rb') as audio_file:
                content: bytes = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="ko-KR"
            )

            loop = asyncio.get_event_loop()
            response: speech.RecognizeResponse = await loop.run_in_executor(
                None,
                partial(
                    self.stt_client.recognize,
                    config=config,
                    audio=audio
                )
            )

            if not response.results:
                return ""

            transcript: str = response.results[0].alternatives[0].transcript.strip()
            return transcript

        except Exception as e:
            logger.error(f"Speech-to-Text 변환 오류: {e}")
            return ""


class AIChatGUI(QMainWindow):
    """AI Chat GUI 애플리케이션 클래스."""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Magic Conch Shell")
        self.setGeometry(100, 100, 800, 800)
        self.setStyleSheet("background-color: #ffffff;")
        self.setWindowIcon(QIcon("icon.png"))

        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
        self.last_ai_response: Optional[str] = None

        self.setup_stt()
        self.init_tts_client()
        self.load_sound_effect()

        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self.loop_thread: threading.Thread = threading.Thread(target=self.start_loop, daemon=True)
        self.loop_thread.start()

        self.aiohttp_session: aiohttp.ClientSession = aiohttp.ClientSession(loop=self.loop)

        self.recording: bool = False
        self.frames: list[bytes] = []
        self.stop_event: threading.Event = threading.Event()

        self.speech_queue: Queue = Queue()
        self.speech_thread: threading.Thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()

        self.signals: WorkerSignals = WorkerSignals()

        # 신호 연결
        self.signals.message.connect(self.append_ai_message)
        self.signals.start_processing_image.connect(self.start_processing_image_slot)
        self.signals.start_tts_image.connect(self.start_tts_image_slot)
        self.signals.stop_tts_image.connect(self.stop_tts_image_slot)

        self.tts_manager: TTSManager = TTSManager(self.tts_client, self.signals)
        self.stt_manager: STTManager = STTManager(self.stt_client)

        self.setup_gpio()

        self.create_main_layout()
        self.create_image_frame()
        self.create_chat_area()
        self.create_input_area()

        self.show()

    def setup_gpio(self) -> None:
        if Button is not None:
            try:
                self.button = Button(17)
                # 버튼을 누를 때 녹음 시작, 놓을 때 녹음 중지
                self.button.when_pressed = self.start_recording
                self.button.when_released = self.stop_recording
                logger.info("GPIO 버튼 설정 완료.")
            except Exception as e:
                logger.error(f"GPIO 설정 오류: {e}")
        else:
            logger.info("현재 운영체제에서는 GPIO를 지원하지 않습니다.")

    def start_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def create_main_layout(self) -> None:
        self.central_widget: QWidget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout: QVBoxLayout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

        top_frame: QFrame = QFrame()
        top_layout: QHBoxLayout = QHBoxLayout(top_frame)

        header_label: QLabel = QLabel("Magic Conch Shell")
        header_font: QFont = QFont("Helvetica", 24, QFont.Weight.Bold)
        header_label.setFont(header_font)
        header_label.setStyleSheet("color: #2c3e50;")

        self.status_label: QLabel = QLabel("Status: Waiting")
        status_font: QFont = QFont("Helvetica", 12)
        self.status_label.setFont(status_font)
        self.status_label.setStyleSheet("color: #16a085;")

        top_layout.addWidget(header_label)
        top_layout.addStretch()
        top_layout.addWidget(self.status_label)

        self.main_layout.addWidget(top_frame)

    def create_image_frame(self) -> None:
        image_frame: QFrame = QFrame()
        image_layout: QVBoxLayout = QVBoxLayout(image_frame)

        try:
            original_image: Optional[QPixmap] = self.load_and_resize_image("co1.jpg", (500, 348))
            self.original_image: Optional[QPixmap] = original_image

            processing_movie: QMovie = QMovie("processing.gif")
            if not processing_movie.isValid():
                raise FileNotFoundError("Processing GIF 파일이 유효하지 않습니다: processing.gif")
            self.processing_movie: QMovie = processing_movie

            tts_image: Optional[QPixmap] = self.load_and_resize_image("co2.jpg", (500, 348))
            self.tts_image: Optional[QPixmap] = tts_image

            self.image_label: QLabel = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("background-color: #ffffff;")
            self.image_label.setPixmap(self.original_image)

            image_layout.addWidget(self.image_label)
        except Exception as e:
            logger.error(f"이미지를 불러오는 중 오류 발생: {e}")
            self.image_label = QLabel("이미지를 불러올 수 없습니다.")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet("background-color: #ffffff;")
            image_layout.addWidget(self.image_label)

        self.main_layout.addWidget(image_frame)

    def load_and_resize_image(self, path: str, size: tuple[int, int]) -> Optional[QPixmap]:
        """이미지를 로드하고 지정된 크기로 리사이즈."""
        try:
            pil_image: Image.Image = Image.open(path)
            pil_image = pil_image.resize(size, Image.LANCZOS)
            return self.pil_image_to_qpixmap(pil_image)
        except Exception as e:
            logger.error(f"이미지 로드 오류 ({path}): {e}")
            return None

    def pil_image_to_qpixmap(self, pil_image: Image.Image) -> QPixmap:
        rgb_image: Image.Image = pil_image.convert("RGB")
        data: bytes = rgb_image.tobytes("raw", "RGB")
        qimage: QImage = QImage(data, pil_image.width, pil_image.height, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage)

    def set_image(self, image: QPixmap) -> None:
        """이미지를 설정하고, 현재 실행 중인 애니메이션을 중지."""
        if hasattr(self, "processing_movie") and self.processing_movie.state() == QMovie.MovieState.Running:
            self.processing_movie.stop()
        self.image_label.setPixmap(image)

    def create_chat_area(self) -> None:
        chat_frame: QFrame = QFrame()
        chat_layout: QVBoxLayout = QVBoxLayout(chat_frame)

        self.chat_area: QTextEdit = QTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setFont(QFont("Helvetica", 12))
        self.chat_area.setStyleSheet("""
            QTextEdit {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: none;
                padding: 10px;
                border-radius: 10px;
            }
        """)

        chat_layout.addWidget(self.chat_area)
        self.main_layout.addWidget(chat_frame)

    def create_input_area(self) -> None:
        input_frame: QFrame = QFrame()
        input_layout: QHBoxLayout = QHBoxLayout(input_frame)

        self.user_input: QLineEdit = QLineEdit()
        self.user_input.setFont(QFont("Helvetica", 12))
        self.user_input.setFixedHeight(40)
        self.user_input.setPlaceholderText("메시지를 입력하세요...")
        self.user_input.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                padding: 10px;
                border-radius: 5px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
        """)
        self.user_input.returnPressed.connect(self.send_message)

        self.send_button: QPushButton = QPushButton("제출")
        self.send_button.setFixedSize(80, 40)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: #ffffff;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.send_button.clicked.connect(self.send_message)

        self.voice_button: QPushButton = QPushButton("음성 입력")
        self.voice_button.setFixedSize(100, 40)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #1abc9c;
                color: #ffffff;
                border: none;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #16a085;
            }
        """)
        self.voice_button.pressed.connect(self.start_recording)
        self.voice_button.released.connect(self.stop_recording)

        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.voice_button)

        self.main_layout.addWidget(input_frame)

    def setup_stt(self) -> None:
        try:
            self.stt_client: speech.SpeechClient = speech.SpeechClient()
            logger.info("Google Cloud Speech-to-Text 클라이언트 초기화 완료.")
        except Exception as e:
            logger.error(f"Google Cloud STT 초기화 오류: {e}")
            sys.exit(1)

    def init_tts_client(self) -> None:
        try:
            self.tts_client: texttospeech.TextToSpeechClient = texttospeech.TextToSpeechClient()
            logger.info("Google Cloud Text-to-Speech 클라이언트 초기화 완료.")
        except Exception as e:
            logger.error(f"Google TTS 초기화 오류: {e}")
            sys.exit(1)

    def load_sound_effect(self) -> None:
        try:
            sound_effect_path: str = "bogle.wav"
            if not os.path.exists(sound_effect_path):
                raise FileNotFoundError(f"효과음 파일이 존재하지 않습니다: {sound_effect_path}")
            self.sound_effect: Optional[sa.WaveObject] = sa.WaveObject.from_wave_file(sound_effect_path)
            logger.info("효과음 파일 로드 완료.")
        except Exception as e:
            logger.error(f"효과음 파일 로드 오류: {e}")
            self.sound_effect = None

    def speech_worker(self) -> None:
        while True:
            item: Optional[str] = self.speech_queue.get()
            if item is None:
                break
            try:
                if item == "__PLAY_SOUND_EFFECT__":
                    if self.sound_effect:
                        play_obj: sa.PlayObject = self.sound_effect.play()
                        logger.info("효과음 재생 시작.")
                        play_obj.wait_done()
                        logger.info("효과음 재생 완료.")
                else:
                    self.signals.start_tts_image.emit()
                    asyncio.run_coroutine_threadsafe(
                        self.tts_manager.synthesize_and_play(item),
                        self.loop
                    )
            except Exception as e:
                logger.error(f"TTS 재생 중 오류: {e}")
            finally:
                self.speech_queue.task_done()

    def start_processing_image_slot(self) -> None:
        logger.info("이미지 변경: 메시지 제출 시 처리 중 애니메이션 시작")
        if self.processing_movie.state() == QMovie.MovieState.Running:
            self.processing_movie.stop()
        self.image_label.setMovie(self.processing_movie)
        self.processing_movie.start()

    def start_tts_image_slot(self) -> None:
        logger.info("이미지 변경: TTS 시작 시 TTS 이미지로 변경")
        if self.processing_movie.state() == QMovie.MovieState.Running:
            self.processing_movie.stop()
        self.image_label.setPixmap(self.tts_image)

    def stop_tts_image_slot(self) -> None:
        logger.info("이미지 변경: TTS 완료 시 원래 이미지로 복귀")
        self.set_image(self.original_image)

    def send_message(self) -> None:
        try:
            user_text: str = self.user_input.text().strip()
            if not user_text:
                logger.warning("빈 메시지를 보낼 수 없습니다.")
                return

            self.display_message("User", user_text)
            self.user_input.clear()
            self.signals.start_processing_image.emit()
            self.speech_queue.put("__PLAY_SOUND_EFFECT__")
            self.executor.submit(self.process_ai_response, user_text)
        except Exception as e:
            logger.error(f"send_message 메서드에서 오류 발생: {e}")

    def start_recording(self) -> None:
        """음성 녹음 시작"""
        try:
            if not self.recording:
                self.recording = True
                self.frames = []
                self.stop_event.clear()
                self.recording_thread: threading.Thread = threading.Thread(target=self.record_audio, daemon=True)
                self.recording_thread.start()
                self.update_status("Status: Recording...", "#e74c3c")
                logger.info("음성 녹음 시작")
        except Exception as e:
            logger.error(f"start_recording 메서드에서 오류 발생: {e}")

    def stop_recording(self) -> None:
        """음성 녹음 중지"""
        try:
            if self.recording:
                self.recording = False
                self.stop_event.set()
                self.recording_thread.join()
                self.update_status("Status: Processing voice...", "#f39c12")
                logger.info("음성 녹음 중지")
                audio_file_path: str = "voice_input.wav"
                try:
                    with wave.open(audio_file_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                        wf.setframerate(16000)
                        wf.writeframes(b''.join(self.frames))
                    logger.info(f"녹음 완료, 파일 저장됨: {audio_file_path}")
                    self.signals.start_processing_image.emit()
                    self.speech_queue.put("__PLAY_SOUND_EFFECT__")
                    self.executor.submit(self.process_voice_input, audio_file_path)
                except Exception as e:
                    logger.error(f"오디오 파일 저장 중 오류: {e}")
                    self.update_status("Status: Waiting", "#27ae60")
        except Exception as e:
            logger.error(f"stop_recording 메서드에서 오류 발생: {e}")

    def record_audio(self) -> None:
        """음성 녹음 스레드"""
        chunk_size: int = 1024
        sample_format: int = pyaudio.paInt16
        channels: int = 1
        sample_rate: int = 16000

        audio_interface: pyaudio.PyAudio = pyaudio.PyAudio()

        try:
            stream = audio_interface.open(
                format=sample_format,
                channels=channels,
                rate=sample_rate,
                frames_per_buffer=chunk_size,
                input=True
            )
            while not self.stop_event.is_set():
                data: bytes = stream.read(chunk_size, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            logger.error(f"녹음 중 오류 발생: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio_interface.terminate()

    def process_voice_input(self, audio_file_path: str) -> None:
        """음성 입력 처리"""
        asyncio.run_coroutine_threadsafe(
            self.handle_voice_input(audio_file_path),
            self.loop
        )

    async def handle_voice_input(self, audio_file_path: str) -> None:
        """음성 파일을 텍스트로 변환하고 AI 응답 처리"""
        try:
            if not os.path.exists(audio_file_path):
                raise Exception("음성 파일이 생성되지 않았습니다.")
            if os.path.getsize(audio_file_path) == 0:
                raise Exception("음성 파일이 비어 있습니다.")

            logger.info(f"녹음된 파일 크기: {os.path.getsize(audio_file_path)} bytes")
            self.update_status("Status: Recognizing voice...", "#2980b9")
            logger.info("Speech-to-Text 변환 시작")

            user_text: str = await self.stt_manager.transcribe_audio(audio_file_path)

            if not user_text:
                logger.info("음성 인식 실패, 사용자에게 다시 요청.")
                await self.tts_manager.synthesize_and_play("죄송합니다, 이해하지 못했습니다. 다시 말씀해 주세요.")
                self.update_status("Status: Waiting", "#27ae60")
                return

            logger.info(f"인식된 텍스트: {user_text}")
            self.display_message("User (Voice)", user_text)
            self.signals.start_processing_image.emit()
            self.speech_queue.put("__PLAY_SOUND_EFFECT__")
            self.executor.submit(self.process_ai_response, user_text)
        except Exception as e:
            self.update_status("Status: Waiting", "#27ae60")
            logger.error(f"음성 입력 처리 중 오류: {e}")

    def process_ai_response(self, user_input: str) -> None:
        """AI 응답 처리"""
        asyncio.run_coroutine_threadsafe(
            self.get_and_display_response(user_input),
            self.loop
        )

    async def get_and_display_response(self, user_input: str) -> None:
        """OpenAI API를 사용하여 AI 응답을 가져오고 표시"""
        try:
            ai_response: str = ""
            headers: dict[str, str] = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }

            data: dict[str, Any] = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "너는 '마법의 소라고둥'처럼 행동하는 인공지능 역할을 맡고 있다. "
                            "사용자가 질문을 할 때, 너의 응답은 항상 짧고 단순하며, 약간 모호해야 한다. "
                            "너는 절대 길거나 복잡한 설명을 제공하지 않고, 답변이 최대한 미니멀하게 유지되도록 한다. "
                            "네가 제공하는 답변은 때때로 직관적이지 않을 수 있다. "
                            "하지만 사용자의 질문이 아무리 구체적이거나 긴박하더라도, 너는 항상 차분하고 일관된 태도로 간단히 대답해야 한다. "
                            "너는 대답할 때 추가적인 정보를 제공하지 않으며, 이유나 배경 설명도 주지 않는다. "
                            "모든 대답은 명확하게 끝마쳐야 하고, 질문의 맥락이나 세부 사항을 언급하지 않는다. "
                            "너는 최대한 짧고 간결하게 대답해야 한다. "
                            "존댓말은 사용하지 않는다. 대답은 항상 간단하게 유지한다.'다.'로 끝나는 말을 사용 하지 않는다. "
                        )
                    },
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 500,
                "n": 1,
                "stop": None,
                "temperature": 0.7,
                "stream": True
            }

            async with self.aiohttp_session.post(
                OPENAI_API_URL, headers=headers, json=data
            ) as response:
                if response.status != 200:
                    error_info: str = await response.text()
                    raise Exception(f"OpenAI API 오류 {response.status}: {error_info}")

                async for line in response.content:
                    if line:
                        decoded_line: str = line.decode('utf-8').strip()
                        if decoded_line.startswith("data: "):
                            data_str: str = decoded_line[len("data: "):]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json: dict[str, Any] = json.loads(data_str)
                                delta: dict[str, Any] = data_json.get('choices', [])[0].get('delta', {})
                                content: str = delta.get('content', '')
                                if content:
                                    ai_response += content
                            except Exception as e:
                                logger.error(f"스트리밍 데이터 처리 오류: {e}")

            if ai_response:
                self.stop_sound_effect()
                self.speech_queue.put(ai_response)
                self.signals.message.emit(ai_response)

            self.last_ai_response = ai_response
            self.update_status("Status: Waiting", "#27ae60")

        except Exception as e:
            logger.error(f"AI 응답 가져오는 중 오류: {e}")
            self.update_status("Status: Waiting", "#27ae60")

    def append_ai_message(self, message: str) -> None:
        self.display_message("AI", message)

    def display_message(self, sender: str, message: str) -> None:
        sender_style: str = f"<span style='color:#2c3e50; font-weight:bold;'>{sender}: </span>"
        message_style: str = f"<span style='color:#2c3e50;'>{message}</span>"
        self.chat_area.append(f"{sender_style}{message_style}<br>")
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )

    def update_status(self, text: str, color: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color};")

    def stop_sound_effect(self) -> None:
        with self.tts_manager.play_obj_lock:
            if self.tts_manager.current_play_obj and self.tts_manager.current_play_obj.is_playing():
                self.tts_manager.current_play_obj.stop()
                self.tts_manager.current_play_obj = None
                logger.info("효과음 재생 중지됨.")

    def stop(self) -> None:
        try:
            self.speech_queue.put(None)
            self.speech_thread.join()
            self.stop_sound_effect()

            close_future: asyncio.Future = asyncio.run_coroutine_threadsafe(
                self.aiohttp_session.close(),
                self.loop
            )
            close_future.result(timeout=10)

            self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop_thread.join()

            if Button is not None and hasattr(self, 'button'):
                self.button.close()
                logger.info("GPIO 클린업 완료.")
        except Exception as e:
            logger.error(f"stop 메서드에서 오류 발생: {e}")

    def closeEvent(self, event: Any) -> None:
        self.stop()
        event.accept()

    def speech_worker(self) -> None:
        while True:
            item: Optional[str] = self.speech_queue.get()
            if item is None:
                break
            try:
                if item == "__PLAY_SOUND_EFFECT__":
                    if self.sound_effect:
                        play_obj: sa.PlayObject = self.sound_effect.play()
                        logger.info("효과음 재생 시작.")
                        play_obj.wait_done()
                        logger.info("효과음 재생 완료.")
                else:
                    self.signals.start_tts_image.emit()
                    asyncio.run_coroutine_threadsafe(
                        self.tts_manager.synthesize_and_play(item),
                        self.loop
                    )
            except Exception as e:
                logger.error(f"TTS 재생 중 오류: {e}")
            finally:
                self.speech_queue.task_done()

    def handle_gpio_button_pressed(self) -> None:
        """
        GPIO 버튼이 눌렸을 때 호출되는 메서드.
        예시로 음성 녹음을 시작하도록 설정.
        """
        logger.info("GPIO 버튼을 통한 음성 녹음 시작 요청.")
        self.start_recording()


def main() -> None:
    def exception_hook(exctype: type, value: BaseException, tb: Any) -> None:
        logger.error("Uncaught exception", exc_info=(exctype, value, tb))
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = exception_hook

    app: QApplication = QApplication(sys.argv)
    gui: AIChatGUI = AIChatGUI()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
