import os
import shutil

# Path to your original dataset
SOURCE_DIR = "Data/AUDIO"

# Path where organized data will go
TARGET_DIR = "Data/organized_audio"

# Emotion mapping
emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

def organize_dataset():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    for actor in os.listdir(SOURCE_DIR):
        actor_path = os.path.join(SOURCE_DIR, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):
            if not file.endswith(".wav"):
                continue

            parts = file.split("-")
            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                continue

            emotion = emotion_map[emotion_code]

            emotion_folder = os.path.join(TARGET_DIR, emotion)
            os.makedirs(emotion_folder, exist_ok=True)

            src = os.path.join(actor_path, file)
            dst = os.path.join(emotion_folder, file)

            shutil.copy(src, dst)

    print("✅ Dataset organized successfully!")

if __name__ == "__main__":
    organize_dataset()