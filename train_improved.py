
from ultralytics import YOLO

# Start from pretrained YOLO weights
model = YOLO("yolo11n.pt")

# Train an improved model
model.train(
    data="construction_ppe_yolo/data.yaml",
    epochs=80,
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    patience=20,
    pretrained=True,
    project=r"C:\Users\ladhr\Industrial-PPE-Detection\runs\ppe_improved",
    name="train"
)
