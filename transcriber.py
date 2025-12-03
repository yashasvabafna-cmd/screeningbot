from openai import OpenAI
import io
from scipy.io.wavfile import write
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
openai_client = OpenAI(api_key=api_key)

def transcribe_audio(audio_data, fs):
    """
    Transcribes audio data using OpenAI's Whisper model.
    
    Args:
        audio_data (numpy.ndarray): Audio data as a numpy array.
        fs (int): Sampling rate.
        
    Returns:
        str: The transcribed text.
    """
    # Convert to WAV in memory
    audio_bytes = io.BytesIO()
    write(audio_bytes, fs, audio_data)
    audio_bytes.seek(0)
    audio_bytes.name = "audio.wav"

    try:
        # Whisper transcription
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            language="en"   # Force English transcription
        )
        return transcription.text
    except Exception as e:
        return f"Error during transcription: {e}"
