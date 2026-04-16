import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)            # remove mentions/hashtags
    text = re.sub(r"[^a-z0-9\s!?.,']", "", text)    # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text