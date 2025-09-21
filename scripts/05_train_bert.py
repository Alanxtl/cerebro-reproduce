import json, pathlib, random
import sys
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

DATA = pathlib.Path(sys.argv[1])  # each line: {"pkg":..., "sequence_text":..., "label":0/1}
rows = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]

y = [r["label"] for r in rows]
train_rows, temp_rows, y_train, y_temp = train_test_split(
    rows, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

y_temp_list = [r["label"] for r in temp_rows]
eval_rows, test_rows, y_eval, y_test = train_test_split(
    temp_rows, y_temp_list, test_size=0.5, random_state=42, shuffle=True, stratify=y_temp_list
)

print(f"Split sizes => train: {len(train_rows)}, eval: {len(eval_rows)}, test: {len(test_rows)}")

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

def enc(batch):
    enc_out = tok(batch["sequence_text"], truncation=True, padding="max_length", max_length=256)
    enc_out["labels"] = batch["label"]
    return enc_out

train_ds = Dataset.from_list(train_rows).map(enc, batched=True)
eval_ds  = Dataset.from_list(eval_rows).map(enc, batched=True)
test_ds  = Dataset.from_list(test_rows).map(enc, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def metrics(p):
    preds = p.predictions.argmax(-1)
    y = p.label_ids
    acc = accuracy_score(y, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y, preds, average='binary', zero_division=0)
    return {"accuracy":acc, "precision":pr, "recall":rc, "f1":f1}

args = TrainingArguments(
    output_dir="./outputs/models/bert",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=2e-5,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, tokenizer=tok, compute_metrics=metrics)
trainer.train()
trainer.evaluate()
trainer.save_model("./outputs/models/bert-final")

preds_output = trainer.predict(test_ds)
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=-1)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))