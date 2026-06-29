"""
confusion.py
Confusion matrix: adaptive ML selector chosen method vs true best method.
Uses chosen_method column logged directly during evaluate_adaptive.py.

Outputs: results/plots/confusion_matrix.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

os.makedirs("results/plots", exist_ok=True)

df = pd.read_csv("results/adaptive_eval.csv")
df = df.dropna(subset=["chosen_method", "best_baseline"])

y_true = df["best_baseline"]
y_pred = df["chosen_method"]

labels = sorted(set(y_true) | set(y_pred))

cm = confusion_matrix(y_true, y_pred, labels=labels)

print("=== CONFUSION MATRIX ===")
print(f"Labels: {labels}\n")
print(cm)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

accuracy = (y_true == y_pred).mean()
print(f"\nSelector accuracy (exact match): {accuracy:.1%}")

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels,
)
plt.xlabel("Predicted Method")
plt.ylabel("True Best Method")
plt.title(f"Confusion Matrix — Adaptive ML Selector\nExact accuracy: {accuracy:.1%}")
plt.tight_layout()
plt.savefig("results/plots/confusion_matrix.png", dpi=300)
plt.show()
print("Saved: results/plots/confusion_matrix.png")