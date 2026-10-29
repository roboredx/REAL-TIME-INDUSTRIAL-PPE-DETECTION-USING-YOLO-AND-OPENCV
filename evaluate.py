
from ultralytics import YOLO

# Load your trained PPE detection model
model = YOLO(
    r"runs/detect/runs/ppe/train/weights/best.pt"
)

# Evaluate using the test dataset
metrics = model.val(
    data="construction_ppe_yolo/data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    device="cpu"
)

# Display evaluation results
print("\n========== PPE MODEL EVALUATION ==========")
print(f"Precision  : {metrics.box.mp:.4f}")
print(f"Recall     : {metrics.box.mr:.4f}")
print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")
print("==========================================")
