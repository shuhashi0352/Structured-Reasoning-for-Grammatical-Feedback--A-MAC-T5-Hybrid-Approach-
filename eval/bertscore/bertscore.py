import json
from bert_score import score
import matplotlib.pyplot as plt

# Load data
with open("PATH TO MAC-T5 GENERATED FEEDBACK") as f:
    mac_t5_data = json.load(f)
with open("PATH TO T5 GENERATED FEEDBACK") as f:
    t5_data = json.load(f)

# Extract predictions and references
mac_cands = [ex["generated_feedback"].strip() for ex in mac_t5_data]
mac_refs = [ex["ref"].strip() for ex in mac_t5_data]

t5_cands = [ex["generated_feedback"].strip() for ex in t5_data]
t5_refs = [ex["ref"].strip() for ex in t5_data]

# Compute BERTScores
P_mac, R_mac, F1_mac = score(mac_cands, mac_refs, lang="en", verbose=True)
P_t5, R_t5, F1_t5 = score(t5_cands, t5_refs, lang="en", verbose=True)

# Averages
mac_scores = {
    "Precision": P_mac.mean().item(),
    "Recall": R_mac.mean().item(),
    "F1": F1_mac.mean().item()
}

t5_scores = {
    "Precision": P_t5.mean().item(),
    "Recall": R_t5.mean().item(),
    "F1": F1_t5.mean().item()
}

# Write results to a text file
with open("twoshot-bertscore.txt", "w") as f:
    f.write("MAC-T5 (two-shot) BERTScore:\n")
    for metric, score_val in mac_scores.items():
        f.write(f"{metric}: {score_val:.4f}\n")
    f.write("\nT5 (two-shot) BERTScore:\n")
    for metric, score_val in t5_scores.items():
        f.write(f"{metric}: {score_val:.4f}\n")

# Visualization
labels = list(mac_scores.keys())
mac_vals = list(mac_scores.values())
t5_vals = list(t5_scores.values())

x = range(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar([p - width/2 for p in x], mac_vals, width=width, label="MAC-T5")
plt.bar([p + width/2 for p in x], t5_vals, width=width, label="T5")
plt.xticks(ticks=x, labels=labels)
plt.ylabel("BERTScore")
plt.title("BERTScore Comparison: MAC-T5 vs T5 (two-shot)")
plt.legend()
plt.tight_layout()
plt.savefig("twoshot-bertscore.png")
plt.show()
