
from ultralytics import YOLO

model = YOLO(
    r"C:\Users\ladhr\Industrial-PPE-Detection\runs\ppe_improved\train\weights\best.pt"
)

# Use validation data while improving the model
metrics = model.val(
    data="construction_ppe_yolo/data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device="cpu"
)

print("\n===== IMPROVED MODEL RESULTS =====")
print(f"Precision : {metrics.box.mp:.4f}")
print(f"Recall    : {metrics.box.mr:.4f}")
print(f"mAP50     : {metrics.box.map50:.4f}")
print(f"mAP50-95  : {metrics.box.map:.4f}")
