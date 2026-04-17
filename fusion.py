TEXT_TO_FACE = {
    "admiration": "happy", "amusement": "happy", "approval": "happy",
    "caring": "happy", "desire": "happy", "excitement": "happy",
    "gratitude": "happy", "joy": "happy", "love": "happy",
    "optimism": "happy", "pride": "happy", "relief": "happy",

    "anger": "angry", "annoyance": "angry", "disapproval": "angry",

    "disgust": "disgust",

    "fear": "fear", "nervousness": "fear",

    "sadness": "sad", "disappointment": "sad",
    "embarrassment": "sad", "grief": "sad", "remorse": "sad",

    "surprise": "surprise", "realization": "surprise",

    "confusion": "neutral", "curiosity": "neutral", "neutral": "neutral"
}

VOICE_TO_FACE = {
    "Neutral 😐": "neutral",
    "Happy 😄": "happy",
    "Sad 😢": "sad",
    "Angry 😠": "angry"
}

def fuse_all(face_emotion, face_conf,
             text_emotion, text_conf,
             voice_emotion, voice_conf):

    mapped_text = TEXT_TO_FACE.get(text_emotion, "neutral")
    mapped_voice = VOICE_TO_FACE.get(voice_emotion, "neutral")

    scores = {}

    scores[face_emotion] = face_conf * 0.5
    scores[mapped_text] = scores.get(mapped_text, 0) + text_conf * 0.3
    scores[mapped_voice] = scores.get(mapped_voice, 0) + voice_conf * 0.2

    final = max(scores, key=scores.get)

    return final, scores