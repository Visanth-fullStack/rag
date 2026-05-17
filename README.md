# Interview Agent - Live Transcriber

A Windows desktop application (WinUI 3) that transcribes audio from both your microphone and system audio in real-time using a local Whisper model.

## Features
- **Real-time Transcription**: Powered by `faster-whisper` (WhisperX compatible).
- **System Audio Capture**: Capture sound from videos, meetings, or games (Loopback).
- **Microphone Capture**: Standard voice input.
- **Local Processing**: Your audio never leaves your machine.

## Prerequisites
1. **Python 3.10+**: For the transcription backend.
2. **.NET 8 SDK**: For the WinUI 3 frontend.
3. **FFmpeg**: Must be installed and available in your PATH.
4. **NVIDIA GPU (Optional)**: Highly recommended for faster processing (requires CUDA).

## Setup & Run

### 1. Backend (Python)
- Run `.\setup_backend.bat` to create a virtual environment and install dependencies.
- Run `.\run_backend.bat` to start the transcription service. It will listen on `ws://localhost:8000`.

### 2. Frontend (WinUI 3)
- Open `src/InterviewAgent.App/InterviewAgent.App.csproj` in **Visual Studio 2022**.
- Build and Run.
- **Note**: Ensure the Backend is running before clicking "Start Transcription".

## Technical Architecture
- **Frontend**: C# WinUI 3 app using `NAudio` for WASAPI capture and resampling.
- **Backend**: FastAPI server wrapping `faster-whisper` for low-latency inference.
- **Communication**: WebSocket protocol for binary audio streaming.

## Future Improvements
- Add speaker diarization (using WhisperX features).
- Add history log saving.
- Add support for different Whisper model sizes (Tiny, Small, Medium, Large).

