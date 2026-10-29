
from ultralytics import YOLO

# Load the trained SH17 model
model = YOLO(r"runs\detect\runs\ppe_sh17\train\weights\best.pt")

# Evaluate on the test dataset
metrics = model.val(
    data=r"ppe_sh17_yolo\data.yaml",
    split="test",
    imgsz=640,
    batch=4,
    workers=0,
    device="cpu"
)

print("\n===== SH17 PPE MODEL TEST RESULTS =====")
print(f"Precision:  {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
print(f"Recall:     {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
print(f"mAP50:      {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
print(f"mAP50-95:   {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
