import asyncio
from openai import AsyncOpenAI
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
client = AsyncOpenAI(api_key=api_key)

async def _generate_speech(text):
    """
    Generate speech from text using OpenAI's TTS API.
    
    Args:
        text (str): The text to convert to speech.
        
    Returns:
        str: Path to the temporary audio file.
    """
    # Create a temporary file for the audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    
    # Generate speech
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=text,
        instructions="Speak in a warm and friendly tone.",
        response_format="wav",
    ) as response:
        # Write the audio stream to the file
        async for chunk in response.iter_bytes(chunk_size=1024):
            temp_file.write(chunk)
    
    temp_file.close()
    return temp_file.name

def text_to_speech(text):
    """
    Convert text to speech and return the audio file path.
    
    Args:
        text (str): The text to convert to speech.
        
    Returns:
        str: Path to the audio file.
    """
    # Run the async function in a new event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(_generate_speech(text))
