import whisper
import sounddevice as sd
import numpy as np
import tempfile
import time
import os
import scipy.io.wavfile as wav

# Load model
model = whisper.load_model("base")

SAMPLE_RATE = 16000
DURATION = 5  # seconds per segment

def record_segment():
    print("🎤 Recording... Speak now!")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    return np.squeeze(audio)

def transcribe_segment(audio_data):
    # Save temporary wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmp_path = tmpfile.name
        wav.write(tmp_path, SAMPLE_RATE, audio_data)
    
    # Transcribe
    result = model.transcribe(tmp_path)
    
    try:
        os.remove(tmp_path)
    except PermissionError:
        time.sleep(1)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    return result["text"]

print("🚀 Whisper real-time test started.")
print("Speak something every few seconds. Press Ctrl+C to stop.\n")

try:
    while True:
        audio_data = record_segment()
        text = transcribe_segment(audio_data)
        if text.strip():
            print(f"🗣️ You said: {text}")
        else:
            print("No speech detected.")
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopped.")
