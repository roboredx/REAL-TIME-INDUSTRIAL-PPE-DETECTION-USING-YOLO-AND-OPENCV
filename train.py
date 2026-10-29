
from ultralytics import YOLO

# Load a small pretrained YOLO11 model
model = YOLO("yolo11n.pt")

# Train on your downloaded PPE dataset
model.train(
    data="construction_ppe_yolo/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu",
    project="runs/ppe",
    name="train"
)

print("Training finished!")
