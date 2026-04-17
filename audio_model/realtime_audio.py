import os
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

print("⏳ Loading model... please wait")
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model_fast")

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()
print("✅ Model loaded successfully!")

labels_map = {
    0: "Neutral 😐",
    1: "Happy 😄",
    2: "Sad 😢",
    3: "Angry 😠"
}

SAMPLE_RATE = 16000
DURATION = 3


# 🎤 Record audio
def record_audio():
    print("\n🎤 Speak now...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    print("⏳ Processing...")
    return audio.flatten()


# 🤖 Predict emotion
def predict(audio):
    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs.input_values)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    return labels_map[pred], confidence


# 🔗 Final function for fusion
def get_audio_emotion():
    audio = record_audio()
    emotion, confidence = predict(audio)
    return emotion, confidence