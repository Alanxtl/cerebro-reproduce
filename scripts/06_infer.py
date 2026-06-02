import json, sys, pathlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    confusion_matrix,
)
from tqdm import tqdm

MODEL_DIR = "./outputs/models/bert-final"
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

in_jsonl = pathlib.Path(sys.argv[1])

def get_types(o):
    mt = o.get("malicious_types")
    if isinstance(mt, list) and mt:
        return [str(x) for x in mt if str(x).strip()]
    tb = o.get("type_bucket")
    if tb:
        return [str(tb)]
    return []


y_true, y_pred = [], []
type_stats = {}
unknown_mal = 0
lines = in_jsonl.read_text(encoding="utf-8").splitlines()
for line in tqdm(lines, desc="Inferencing"):
    o = json.loads(line)
    t = tok(
        o["sequence_text"],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        logits = model(**t).logits
        prob = logits.softmax(-1)[0].tolist()
    pred = int(torch.argmax(logits, dim=-1).item())

    y_true.append(int(o["label"]))
    y_pred.append(pred)
    if int(o["label"]) == 1:
        types = get_types(o)
        if not types:
            unknown_mal += 1
        for tp in types:
            stat = type_stats.setdefault(tp, {"total": 0, "hit": 0})
            stat["total"] += 1
            if pred == 1:
                stat["hit"] += 1

    # print(json.dumps({
    #     "pkg": o.get("pkg"),
    #     "label": o["label"],
    #     "pred": pred,
    #     "prob_benign": prob[0],
    #     "prob_malicious": prob[1]
    # }))

# ===== 计算指标 =====
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
cm = confusion_matrix(y_true, y_pred)

report = classification_report(y_true, y_pred)

print("\n=== Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("Confusion matrix:\n", cm)
print("\nClassification Report:\n", report)

if type_stats:
    print("\n=== Detection Rate by Malicious Type ===")
    items = sorted(type_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for tp, stat in items:
        total = stat["total"]
        hit = stat["hit"]
        rate = (hit / total) if total else 0.0
        print(f"{tp}: {rate:.4f} ({hit}/{total})")
if unknown_mal:
    print(f"\nMalicious samples missing type info: {unknown_mal}")
