# Multimodal Emotion AI Project

An AI-based system that detects human emotions using **multiple modalities** — audio, facial expressions, and text — and combines them to produce a more accurate and robust prediction.

---

## 🚀 Features

* 🎧 **Audio Emotion Recognition**

  * Uses Wav2Vec2 for speech-based emotion detection
  * Supports emotions: **Happy, Sad, Angry, Neutral**
  * Real-time audio recording support

* 😀 **Face Emotion Recognition**

  * Uses deep learning (ResNet-based model)
  * Detects emotions from facial expressions

* 💬 **Text Emotion Detection**

  * NLP-based emotion classification from user input

* 🔗 **Fusion Module**

  * Combines outputs from all modalities
  * Improves overall accuracy

---

## 🧠 Tech Stack

* Python
* PyTorch
* Transformers (Wav2Vec2)
* OpenCV
* Librosa
* SoundDevice

---

## 📁 Project Structure

```
project/
│
├── audio_model/
│   ├── realtime_audio.py
│   ├── train_audio_model.py
│   ├── organize_dataset.py
│   └── emotion_model_fast/   (download required)
│
├── face_model/
├── text_model/
├── fusion.py
├── app.py
└── README.md
```

---

## ⚠️ Audio Model Setup (IMPORTANT)

The trained audio model is **not included** in this repository due to GitHub size limits.

### 📥 Download Model

Download from:
👉 https://drive.google.com/drive/folders/1AYoAJMf2jvOOVNOXRVbAAMGMR7XljtzG?usp=drive_link

---

### 📂 After Download

1. Extract (if zipped)
2. Place the folder inside:

```
audio_model/emotion_model_fast/
```

---

### ✅ Final Structure Should Be:

```
audio_model/
└── emotion_model_fast/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.json
    └── ...
```

---

## ▶️ How to Run

### 🔹 1. Install dependencies

```
pip install -r requirements.txt
```

---

### 🔹 2. Run Audio Module (Example)

```
python audio_model/realtime_audio.py
```

---

### 🔹 3. Run Full System

```
python app.py
```

---

## 🧪 Demo Flow

1. User speaks / inputs text / shows face
2. Each module predicts emotion
3. Fusion module combines results
4. Final emotion is displayed

---

## 📊 Supported Emotions

* 😄 Happy
* 😢 Sad
* 😠 Angry
* 😐 Neutral

---

## 👥 Contributors

* **Naisha Singh** – Audio Emotion Recognition
* **[Teammate Name]** – Face Emotion Module
* **[Teammate Name]** – Text Emotion Module

---

## 💡 Future Improvements

* Real-time UI dashboard
* Multi-language support
* Adaptive AI based on user behavior
* Emotion-aware recommendation system

---

## 📌 Notes

* Dataset and large model files are excluded from GitHub
* Ensure correct folder placement before running
* Designed for educational and demonstration purposes

---

## ⭐ Acknowledgements

* HuggingFace Transformers
* PyTorch
* Open-source emotion datasets

---
