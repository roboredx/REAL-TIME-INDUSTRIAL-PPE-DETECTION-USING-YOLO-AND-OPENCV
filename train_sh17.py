
from ultralytics import YOLO

# Start from your existing trained model
model = YOLO(
    r"runs\ppe_improved\train\weights\best.pt"
)

model.train(
    data=r"ppe_sh17_yolo\data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    project="runs/ppe_sh17",
    name="train"
)
