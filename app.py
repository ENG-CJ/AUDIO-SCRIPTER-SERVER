from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

app = Flask(__name__)
CORS(app)
model = whisper.load_model("base")


@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Whisper API"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"})

    file = request.files['audio']
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    result = model.transcribe(file_path, fp16=False)
    os.remove(file_path)
    return jsonify({"transcription": result['text']})



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
@app.route('/start-live-transcription', methods=['POST'])
def start_live_transcription():
    data = request.json
    duration = int(data.get('duration', 10))  # Duration in seconds, default 10 seconds
    
    total_duration = 0
    chunk_duration = 2  # Transcribe every 2 seconds
    transcribed_text = ""
    
    while total_duration < duration:
        audio_file = record_audio_chunk(chunk_duration)
        text = transcribe_audio_file(audio_file)
        print("Transcribed Text:", text)
        transcribed_text += text + " "
        total_duration += chunk_duration
    
    return jsonify({"transcription": transcribed_text})

# if __name__ == '__main__':
#     os.makedirs("temp", exist_ok=True)
#     app.run(debug=True, port=5000, host='localhost')
