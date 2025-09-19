import json, sys, pathlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_DIR = "./outputs/models/bert-final"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

in_jsonl = pathlib.Path(sys.argv[1])
for line in in_jsonl.read_text(encoding="utf-8").splitlines():
    o = json.loads(line)
    t = tok(o["sequence_text"], return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**t).logits
        prob = logits.softmax(-1)[0].tolist()
    print(json.dumps({"pkg": o.get("pkg"), "prob_benign": prob[0], "prob_malicious": prob[1]}))
