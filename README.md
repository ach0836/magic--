# Magic Conch Shell

**Magic Conch Shell** is an interactive AI-powered chat application that combines voice input, text-to-speech (TTS), speech-to-text (STT), and GPIO (General Purpose Input/Output) functionalities. Designed for both desktop and embedded systems like Raspberry Pi, this application provides a seamless and intuitive user experience, enabling users to communicate with an AI assistant using both text and voice commands.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [GPIO Setup](#gpio-setup)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Graphical User Interface (GUI):** Built with PyQt6 for a user-friendly and responsive interface.
- **Voice Input & Output:** Utilize Google Cloud Speech-to-Text and Text-to-Speech for voice interactions.
- **AI Integration:** Powered by OpenAI's GPT-4 for intelligent and context-aware responses.
- **GPIO Integration:** Control external hardware components like buttons and LEDs (compatible with Raspberry Pi).
- **Caching:** Implements caching to reduce redundant API calls and improve performance.
- **Error Handling:** Robust exception handling to ensure application stability.
- **Logging:** Comprehensive logging for monitoring and debugging purposes.

## Demo

![Magic Conch Shell Demo](images/demo.gif)

*Screenshot showcasing the Magic Conch Shell interface with text and voice interaction capabilities.*

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Git**
- **Google Cloud Account** with Speech-to-Text and Text-to-Speech APIs enabled.
- **OpenAI API Key**
- **Raspberry Pi** (optional, for GPIO functionalities)

### Clone the Repository

```bash
git clone https://github.com/ach0836/magic-conch-shell.git
cd magic-conch-shell
```

### Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Configure Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
```

- **OPENAI_API_KEY:** Your OpenAI API key for accessing GPT-4.
- **GOOGLE_APPLICATION_CREDENTIALS:** Path to your Google Cloud service account JSON file with Speech-to-Text and Text-to-Speech permissions.

### Run the Application

```bash
python magic.py
```

Upon launching, the Magic Conch Shell GUI will appear, allowing you to input text or use voice commands to interact with the AI assistant.

## Configuration

### GPIO Setup (Optional)

If you're running the application on a Raspberry Pi or another device with GPIO capabilities, you can configure a physical button to send messages or trigger actions.

1. **Connect a Button:**
   - Connect one leg of the button to GPIO pin 17.
   - Connect the other leg to the ground (GND).

2. **Modify GPIO Pin (If Necessary):**
   - In `magic.py`, locate the `BUTTON_PIN` variable and change it if you're using a different GPIO pin.

### Audio Configuration

Ensure your microphone and speakers are properly connected and configured on your system. The application uses `pyaudio` for audio input and `simpleaudio` for playback.

## Dependencies

The project relies on the following Python libraries:

- **asyncio:** For asynchronous programming.
- **aiohttp:** For making asynchronous HTTP requests.
- **cachetools:** For implementing caching mechanisms.
- **pyaudio:** For capturing audio input.
- **simpleaudio:** For playing audio output.
- **python-dotenv:** For loading environment variables from `.env` files.
- **google-cloud-speech:** Google Cloud Speech-to-Text client library.
- **google-cloud-texttospeech:** Google Cloud Text-to-Speech client library.
- **Pillow:** For image processing.
- **PyQt6:** For building the GUI.
- **gpiozero:** For GPIO interactions (only on Linux-based systems like Raspberry Pi).

### Installing Dependencies

All dependencies can be installed via `pip` using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
asyncio
aiohttp
cachetools
pyaudio
simpleaudio
python-dotenv
google-cloud-speech
google-cloud-texttospeech
Pillow
PyQt6
gpiozero; sys_platform == 'linux'
```

*Note: The `gpiozero` library is only installed on Linux systems.*

## GPIO Setup

To enable GPIO functionalities on a Raspberry Pi:

1. **Install GPIOZero:**

   ```bash
   sudo apt-get update
   sudo apt-get install python3-gpiozero
   ```

2. **Wiring the Button:**

   - Connect one terminal of the button to GPIO pin 17.
   - Connect the other terminal to the ground (GND).

3. **Run the Application:**

   The application will automatically detect the GPIO button and allow you to interact using the physical button.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or support, please contact:

- **Author:** Ach08
- **Email:** ach0836@example.com
- **GitHub:** [ach0836](https://github.com/ach0836)
