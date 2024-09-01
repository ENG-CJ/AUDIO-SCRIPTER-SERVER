import os
import whisper

print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir())
file_path = os.path.join(os.getcwd(), "test.wav")
try:
    with open(file_path, "rb") as f:
        model = whisper.load_model("base")
        result = model.transcribe(file_path,fp16=False)
        # text = result['chunk']
        print("Transcribed Text:", result)
except FileNotFoundError:
    print("File not found. Please check the path.")
