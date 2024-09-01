import tkinter as tk
from tkinter import filedialog
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import tempfile
import os

model = whisper.load_model("base")

def record_audio(duration=5, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_file.name, fs, recording)
    return temp_file.name

def transcribe_audio_file(audio_file):  
    result = model.transcribe(audio_file, fp16=False)
    os.remove(audio_file)
    return result["text"]

def transcribe_pre_recorded_audio():
    audio_file_path = filedialog.askopenfilename()
    result = model.transcribe(audio_file_path, fp16=False)
    result_label.config(text=result['text'])

def transcribe_live_audio():
    duration = int(duration_entry.get())
    text = transcribe_live_audio_process(duration)
    result_label.config(text=text)

def transcribe_live_audio_process(duration=10):
    total_duration = 0
    chunk_duration = 2
    transcribed_text = ""
    
    while total_duration < duration:
        audio_file = record_audio(chunk_duration)
        text = transcribe_audio_file(audio_file)
        transcribed_text += text + " "
        total_duration += chunk_duration
    
    return transcribed_text

root = tk.Tk()
root.title("Speech-to-Text Interface")

frame = tk.Frame(root)
frame.pack(pady=20)

transcribe_button = tk.Button(frame, text="Transcribe Pre-Recorded Audio", command=transcribe_pre_recorded_audio)
transcribe_button.pack(pady=10)

live_transcribe_button = tk.Button(frame, text="Transcribe Live Audio", command=transcribe_live_audio)
live_transcribe_button.pack(pady=10)

duration_label = tk.Label(frame, text="Enter duration for live transcription (seconds):")
duration_label.pack()

duration_entry = tk.Entry(frame)
duration_entry.pack()

result_label = tk.Label(frame, text="Transcribed Text Will Appear Here", wraplength=400)
result_label.pack(pady=20)

root.mainloop()
