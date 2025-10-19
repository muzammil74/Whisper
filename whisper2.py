import sounddevice as sd
import numpy as np
import whisper
import tempfile
import os
import time
import scipy.io.wavfile as wav

SAMPLE_RATE = 16000
DURATION = 5  # seconds per recording

print("🎙️ Whisper Real-Time Test (Final Stable Version)")
print("🎤 Speak for 5 seconds at a time... Ctrl+C to stop.")

# Load Whisper model
model = whisper.load_model("base")  # use "tiny" for faster testing

def record_audio(duration=5):
    """Record audio for a fixed duration"""
    print("\n🎧 Recording...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording finished.")
    return audio

def transcribe(audio):
    """Save temp WAV, transcribe, and safely delete"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmp_path = tmpfile.name
        wav.write(tmp_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    # Wait a moment to ensure the file is released on Windows
    time.sleep(0.2)
    print("🧠 Transcribing...")
    result = model.transcribe(tmp_path)
    
    # Wait before deleting the file
    time.sleep(0.5)
    try:
        os.remove(tmp_path)
    except PermissionError:
        print("⚠️ Could not delete temp file, skipping cleanup.")
    return result["text"]

try:
    while True:
        audio_data = record_audio(DURATION)
        text = transcribe(audio_data)
        print(f"🗣️ You said: {text.strip()}")
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Exiting. Goodbye!")
