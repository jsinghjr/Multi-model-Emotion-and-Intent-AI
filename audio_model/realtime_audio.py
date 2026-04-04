import sounddevice as sd
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

print("⏳ Loading model... please wait")
processor = Wav2Vec2Processor.from_pretrained("./emotion_model_fast")
model = Wav2Vec2ForSequenceClassification.from_pretrained("./emotion_model_fast")

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

def record_audio():
    print("\n🎤 Speak now...")

    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    print("⏳ Processing...")
    return audio.flatten()

def predict(audio):

    if np.max(np.abs(audio)) != 0:
        audio = audio / np.max(np.abs(audio))

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.no_grad():
        outputs = model(inputs.input_values)

    pred = torch.argmax(outputs.logits).item()

    return labels_map[pred]

while True:
    audio = record_audio()
    emotion = predict(audio)

    print(f"🧠 Detected Emotion: {emotion}")

    cont = input("\nPress Enter to continue or type 'q' to quit: ")
    if cont.lower() == 'q':
        break
    