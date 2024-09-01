import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import tempfile
import os

# Load Whisper model
model = whisper.load_model("base")

def record_audio_chunk(duration=2, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save as temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_file.name, fs, recording)
    return temp_file.name

def transcribe_audio_file(audio_file):
    # Transcribe the audio file
    result = model.transcribe(audio_file, fp16=False)
    
    # Clean up the temporary file
    os.remove(audio_file)
    
    return result["text"]

def transcribe_live_audio(duration=10):
    total_duration = 0
    chunk_duration = 2  # Transcribe every 2 seconds
    transcribed_text = ""
    
    while total_duration < duration:
        audio_file = record_audio_chunk(chunk_duration)
        text = transcribe_audio_file(audio_file)
        print("Transcribed Text:", text)
        transcribed_text += text + " "
        total_duration += chunk_duration
    
    return transcribed_text

# Transcribe live audio
transcribed_text = transcribe_live_audio(duration=10)
print("Final Transcribed Text:", transcribed_text)
