import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from utils.preprocess import clean_text

# ── 1. Load GoEmotions (simplified = 28 emotions collapsed to 6) ──────────
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")
LABELS = dataset["train"].features["labels"].feature.names
NUM_LABELS = len(LABELS)

# ── 2. Tokenizer ──────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(batch):
    texts = [clean_text(t) for t in batch["text"]]
    enc   = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    # GoEmotions can have multiple labels — take the first for simplicity
    enc["labels"] = [lbl[0] if lbl else 0 for lbl in batch["labels"]]
    return enc

tokenized = dataset.map(preprocess, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── 3. Model ──────────────────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS,
)

# ── 4. Training arguments ─────────────────────────────────────────────────
args = TrainingArguments(
    output_dir="./models",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=100,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),   # mixed precision if GPU available
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

trainer.train()
trainer.save_model("./models/emotion_bert")
tokenizer.save_pretrained("./models/emotion_bert")
print("Training complete. Model saved to ./models/emotion_bert")