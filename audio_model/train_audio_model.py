import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "Data", "AUDIO")

SAMPLE_RATE = 16000
EPOCHS = 5
MAX_SAMPLES = 500
BATCH_SIZE = 4

# ---------------------------
# EMOTION MAPPING
# ---------------------------
def get_emotion_from_filename(filename):
    emotion_code = int(filename.split("-")[2])

    emotion_map = {
        1: 0,  # neutral
        3: 1,  # happy
        4: 2,  # sad
        5: 3   # angry
    }

    return emotion_map.get(emotion_code, None)

# ---------------------------
# LOAD MODEL
# ---------------------------
print("⏳ Loading model...")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base-960h",
    num_labels=4
)

# OPTIONAL: comment this if you want more learning
# model.wav2vec2.feature_extractor._freeze_parameters()

device = torch.device("cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# ---------------------------
# LOAD DATA
# ---------------------------
data = []

for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        label = get_emotion_from_filename(file)
        if label is None:
            continue

        file_path = os.path.join(actor_path, file)

        try:
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        except:
            continue

        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # Trim to 3 sec (better context)
        if len(audio) > SAMPLE_RATE * 3:
            audio = audio[:SAMPLE_RATE * 3]

        data.append((audio, label))

        if len(data) >= MAX_SAMPLES:
            break

    if len(data) >= MAX_SAMPLES:
        break

print(f"✅ Loaded {len(data)} samples")

# Shuffle data
np.random.shuffle(data)

# ---------------------------
# TRAIN LOOP
# ---------------------------
for epoch in range(EPOCHS):
    print(f"\n🚀 Starting Epoch {epoch+1}")

    total_loss = 0
    correct = 0
    total = 0

    for i in range(0, len(data), BATCH_SIZE):
        batch = data[i:i+BATCH_SIZE]

        audios = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch]).to(device)

        inputs = processor(
            audios,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(device)

        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Batch {i//BATCH_SIZE + 1} | Loss: {loss.item():.4f}")

    accuracy = (correct / total) * 100

    print(f"🔥 Epoch {epoch+1} Completed | Total Loss: {total_loss:.4f}")
    print(f"✅ Accuracy: {accuracy:.2f}%")

# ---------------------------
# SAVE MODEL
# ---------------------------
SAVE_PATH = os.path.join(BASE_DIR, "emotion_model_fast")

model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)

print("\n🎉 TRAINING COMPLETE + MODEL SAVED!")