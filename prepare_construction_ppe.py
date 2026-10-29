import json
import time
import urllib.request
from pathlib import Path

# Put construction-ppe.ndjson in the same folder as this script.
NDJSON_PATH = Path("construction-ppe.ndjson")
OUTPUT_DIR = Path("construction_ppe_yolo")
RETRY_COUNT = 3

if not NDJSON_PATH.exists():
    raise FileNotFoundError(
        "Could not find construction-ppe.ndjson. "
        "Place this script beside that file and run it again."
    )

# Read the dataset manifest and image records.
dataset_info = None
image_records = []

with NDJSON_PATH.open("r", encoding="utf-8") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)

        if record.get("type") == "dataset":
            dataset_info = record
        elif record.get("type") == "image":
            image_records.append(record)

if not dataset_info or not image_records:
    raise ValueError("The file does not look like the expected PPE dataset NDJSON.")

class_names_dict = dataset_info.get("class_names", {})
class_ids = sorted(int(k) for k in class_names_dict.keys())
class_names = [class_names_dict[str(i)] for i in class_ids]

print(f"Found {len(image_records)} images.")
print("Classes:", ", ".join(class_names))
print(f"Output folder: {OUTPUT_DIR.resolve()}")
print("Downloading images and writing YOLO labels. This may take a while...")

failed = []

for index, record in enumerate(image_records, start=1):
    filename = Path(record.get("file", f"image_{index}.jpg")).name
    split = record.get("split", "train").lower()

    # Map manifest split names to YOLO dataset split names.
    if split not in {"train", "val", "test"}:
        print(f"Skipping unexpected split {split!r} for {filename}")
        continue

    image_dir = OUTPUT_DIR / "images" / split
    label_dir = OUTPUT_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / filename
    label_path = label_dir / (Path(filename).stem + ".txt")

    # The annotations use normalized YOLO-style boxes:
    # [class_id, x_center, y_center, width, height].
    boxes = record.get("annotations", {}).get("boxes", [])
    label_lines = []
    for box in boxes:
        if len(box) != 5:
            continue
        class_id = int(box[0])
        x_center, y_center, width, height = map(float, box[1:])
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    # Download image if it is not already present.
    if not image_path.exists() or image_path.stat().st_size == 0:
        url = record.get("url")
        if not url:
            failed.append((filename, "missing URL"))
            continue

        downloaded = False
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(request, timeout=45) as response:
                    image_bytes = response.read()

                if not image_bytes:
                    raise ValueError("Downloaded file was empty")

                image_path.write_bytes(image_bytes)
                downloaded = True
                break
            except Exception as exc:
                if attempt == RETRY_COUNT:
                    failed.append((filename, str(exc)))
                else:
                    time.sleep(1.5 * attempt)

        if not downloaded:
            continue

    # Write one label file per image; empty labels are valid for images
    # with no annotated objects.
    label_path.write_text("\n".join(label_lines), encoding="utf-8")

    if index % 50 == 0 or index == len(image_records):
        print(f"Processed {index}/{len(image_records)} records")

# Write dataset configuration for Ultralytics YOLO.
yaml_lines = [
    f"path: {OUTPUT_DIR.resolve().as_posix()}",
    "train: images/train",
    "val: images/val",
    "test: images/test",
    "names:",
]
for class_id, name in enumerate(class_names):
    yaml_lines.append(f"  {class_id}: {json.dumps(name)}")

yaml_path = OUTPUT_DIR / "data.yaml"
yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

print("\nPreparation finished.")
print(f"Dataset YAML: {yaml_path.resolve()}")
print(f"Images folder: {(OUTPUT_DIR / 'images').resolve()}")
print(f"Labels folder: {(OUTPUT_DIR / 'labels').resolve()}")

if failed:
    print(f"\nWARNING: {len(failed)} image(s) failed to download.")
    for filename, reason in failed[:15]:
        print(f"  - {filename}: {reason}")
    if len(failed) > 15:
        print(f"  ... and {len(failed) - 15} more")
    print(
        "\nIf errors mention 403, 401, or an expired signature, "
        "download/export a fresh dataset manifest and run this script again."
    )
else:
    print("All listed images were downloaded successfully.")
