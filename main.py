from models.face_model import get_face_emotion
from audio_model.realtime_audio import get_audio_emotion
from text_model.predict import predict_emotion
from fusion import fuse_all

print("🧠 Multimodal Emotion Detection System")

# 1. Take text input
text = input("\nEnter your text: ")

# 2. Run models
face_emotion, face_conf = get_face_emotion()

voice_emotion, voice_conf = get_audio_emotion()

text_result = predict_emotion(text)
text_emotion = text_result["emotion"]
text_conf = text_result["confidence"]

# 3. Fuse
final, scores = fuse_all(
    face_emotion, face_conf,
    text_emotion, text_conf,
    voice_emotion, voice_conf
)

# 4. Output
print("\n===== RESULTS =====")
print(f"Face  : {face_emotion} ({face_conf:.2f})")
print(f"Text  : {text_emotion} ({text_conf:.2f})")
print(f"Voice : {voice_emotion} ({voice_conf:.2f})")

print("\n🔥 FINAL EMOTION:", final)