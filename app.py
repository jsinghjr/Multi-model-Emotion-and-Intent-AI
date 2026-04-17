import streamlit as st
from models.face_model import get_face_emotion
from audio_model.realtime_audio import get_audio_emotion
from text_model.predict import predict_emotion
from fusion import fuse_all

st.title("🧠 Multimodal Emotion Detection")

text = st.text_input("Enter your text")

if st.button("Analyze Emotion"):

    st.write("📷 Capturing face...")
    face_emotion, face_conf = get_face_emotion()

    st.write("🎤 Recording voice...")
    voice_emotion, voice_conf = get_audio_emotion()

    st.write("💬 Analyzing text...")
    text_result = predict_emotion(text)

    final, scores = fuse_all(
        face_emotion, face_conf,
        text_result["emotion"], text_result["confidence"],
        voice_emotion, voice_conf
    )

    st.subheader("Results")
    st.write("Face:", face_emotion)
    st.write("Text:", text_result["emotion"])
    st.write("Voice:", voice_emotion)

    st.success(f"🔥 Final Emotion: {final}")