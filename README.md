# Audio Transcription & Translation App

An elegant application that transcribes audio from various sources (microphone, file upload, YouTube videos) using OpenAI Whisper API and provides translation capabilities.

![App Screenshot](https://img.icons8.com/fluency/96/000000/microphone.png)

## Features

- **Multiple Input Methods**:
  - Record audio directly from your microphone
  - Upload audio/video files (.wav, .mp3, .mp4, etc.)
  - Transcribe audio from YouTube videos by URL

- **Language Support**:
  - Supports 40+ languages for transcription
  - Urdu language prioritized in selection
  - Automatic translation to English using OpenAI

- **Modern UI**:
  - Clean, responsive interface
  - Sidebar navigation
  - Progress indicators for transcription
  - Styled transcript display

- **Well-Structured Code**:
  - Separation of concerns: Core functionality in app.py, UI in streamlit_app.py
  - Modular design for easy maintenance and extension
  - Clean code organization following best practices

## Setup Instructions

### Prerequisites

- Python 3.7+
- Google Cloud Platform account with Speech-to-Text and Translation APIs enabled
- Google Cloud service account key file

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Place your Google Cloud service account JSON key file in the project directory as `service-account.json`.

### Running the Application

Use the streamlit_app.py file to run the application with the improved UI:

```
streamlit run streamlit_app.py
```

## Project Structure

- **app.py**: Contains all core functionality:
  - Audio processing classes and functions
  - Google Cloud API integration
  - File conversion utilities
  - YouTube download functionality

- **streamlit_app.py**: Contains the UI layer:
  - Streamlit interface components
  - Styling and layout
  - User interaction handling
  - Progress display

## Usage Guide

1. **Select Language**: Choose the language of the audio content from the sidebar dropdown.

2. **Choose Translation Option**: Toggle "Translate to English" if you want the transcript translated.

3. **Select Input Method**: Choose between microphone recording, file upload, or YouTube URL.

4. **Process Audio**:
   - For microphone recording: Click "Start Recording", speak, then "Stop Recording"
   - For file upload: Drag and drop or browse to select an audio/video file
   - For YouTube: Enter a valid YouTube video URL and click enter

5. **View Results**: The original transcript and translation (if enabled) will be displayed in styled boxes.

## Technical Details

- Audio chunking for handling longer recordings
- Temporary file management for processing
- Session state management for application flow
- Custom CSS for enhanced UI
- Google Cloud API integration for transcription and translation

## Dependencies

- streamlit
- google-cloud-speech
- google-cloud-translate
- pydub
- numpy
- sounddevice
- yt-dlp
- langdetect
- ffmpeg-python

## License

[MIT License](LICENSE)
