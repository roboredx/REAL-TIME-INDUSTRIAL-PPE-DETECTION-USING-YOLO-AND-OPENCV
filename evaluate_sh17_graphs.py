
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load your trained SH17 model
model = YOLO(r"runs\detect\runs\ppe_sh17\train\weights\best.pt")

# Evaluate on the test set and automatically generate plots
metrics = model.val(
    data=r"ppe_sh17_yolo\data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    plots=True,
    project="runs/ppe_sh17_evaluation",
    name="test"
)

# Output folder
out = Path(metrics.save_dir)
print(f"\nEvaluation results saved to: {out}")

# Get class names and metrics
names = list(model.names.values())
precision = np.array(metrics.box.p) * 100
recall = np.array(metrics.box.r) * 100
map50 = np.array(metrics.box.ap50) * 100
map5095 = np.array(metrics.box.ap) * 100

# GRAPH 1: Compare metrics for every class
x = np.arange(len(names))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - 1.5 * width, precision, width, label="Precision")
ax.bar(x - 0.5 * width, recall, width, label="Recall")
ax.bar(x + 0.5 * width, map50, width, label="mAP50")
ax.bar(x + 1.5 * width, map5095, width, label="mAP50-95")

ax.set_title("SH17 PPE Detection: Per-Class Test Performance")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(out / "per_class_metrics.png", dpi=160)
plt.close(fig)

print("Created:", out / "per_class_metrics.png")
print("\nCheck the same folder for:")
print("- confusion_matrix.png")
print("- confusion_matrix_normalized.png")
print("- PR_curve.png")
print("- F1_curve.png")
print("- P_curve.png")
print("- R_curve.png")
