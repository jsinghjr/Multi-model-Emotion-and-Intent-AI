import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from torchvision import transforms, models

CATEGORIES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def build_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 7)
    )
    return model

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model      = build_model().to(device)
model_path = os.path.join(os.path.dirname(__file__), "emotion_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 🔗 NEW FUNCTION (fusion-ready)
def get_face_emotion():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)
    print("📷 Capturing face...")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "neutral", 0.0

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "neutral", 0.0

    # take first detected face
    (x, y, w, h) = faces[0]

    roi = cv2.resize(gray[y:y+h, x:x+w], (224, 224))
    tensor = transform(roi).unsqueeze(0).to(device)

    with torch.no_grad():
        probs      = torch.softmax(model(tensor), dim=1)[0]
        pred_idx   = probs.argmax().item()
        emotion    = CATEGORIES[pred_idx]
        confidence = probs[pred_idx].item()

    return emotion, confidence


# 🟢 OPTIONAL: keep your old live demo
def analyze_face():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    cap = cv2.VideoCapture(0)
    print("AI Webcam Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (224, 224))
            tensor = transform(roi).unsqueeze(0).to(device)

            with torch.no_grad():
                probs      = torch.softmax(model(tensor), dim=1)[0]
                pred_idx   = probs.argmax().item()
                emotion    = CATEGORIES[pred_idx]
                confidence = probs[pred_idx].item() * 100

            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{emotion} ({confidence:.0f}%)",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow('Emotion AI - Mukul', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # test single prediction
    emotion, conf = get_face_emotion()
    print(f"Detected Emotion: {emotion} ({conf:.2f})")