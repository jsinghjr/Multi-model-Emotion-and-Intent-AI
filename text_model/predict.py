from transformers import pipeline
from text_model.utils.preprocess import clean_text

# Pretrained on GoEmotions — all 28 labels, no training needed
classifier = pipeline(
    "text-classification",
    model="monologg/bert-base-cased-goemotions-original",
    top_k=None,
)

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

def predict_emotion(text: str) -> dict:
    cleaned = clean_text(text)
    results = classifier(cleaned)[0]
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    top = results_sorted[0]
    return {
        "text":       text,
        "emotion":    top["label"],
        "confidence": round(top["score"], 4),
        "top_3":      [(r["label"], round(r["score"], 4)) for r in results_sorted[:3]],
        "all_scores": {r["label"]: round(r["score"], 4) for r in results_sorted},
    }

if __name__ == "__main__":
    samples = [
        "I can't believe they won the championship!",
        "This is so frustrating, nothing is working.",
        "I miss my grandmother every single day.",
        "Oh wow I never thought of it that way!",
        "Thank you so much, this means everything to me.",
    ]
    for s in samples:
        result = predict_emotion(s)
        print(f"\nText      : {result['text']}")
        print(f"Emotion   : {result['emotion']} ({result['confidence']*100:.1f}%)")
        print(f"Top 3     : {result['top_3']}")