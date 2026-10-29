"""
Industrial PPE Detection: benchmark inference speed, memory, and curves.

Run from:
C:\\Users\\ladhr\\Industrial-PPE-Detection

This script:
- Measures model-only inference latency and FPS on the held-out test images.
- Measures end-to-end latency/FPS for image loading + inference + annotation.
- Samples process RAM and GPU VRAM when NVIDIA/torch.cuda is available.
- Saves Ultralytics validation plots and class-specific PR/F1 plots.
- Reads training results.csv to plot training/validation loss curves if available.

Important:
- GPU VRAM is reported as PyTorch allocated/reserved memory, plus device-wide
  memory when supported. These are different measures.
- CPU RAM is process RSS sampled periodically; short-lived peaks can be missed.
- Webcam FPS is not measured here. This script benchmarks test-image processing.
"""

import csv
import os
import time
import threading
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / r"runs\detect\runs\ppe_sh17\train\weights\best.pt"
DATA_YAML = ROOT / r"ppe_sh17_yolo\data.yaml"
OUT_DIR = ROOT / "benchmark_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMGSZ = 640
BATCH = 1                 # batch=1 is most representative for live video
DEVICE = 0 if torch.cuda.is_available() else "cpu"
WARMUP_RUNS = 10
MAX_IMAGES = None         # set to e.g. 100 for a quick benchmark; None = all test images
CONF = 0.25

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

model = YOLO(str(MODEL_PATH))
process = psutil.Process(os.getpid())

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_ram_mb():
    return process.memory_info().rss / (1024 ** 2)

def get_vram_mb():
    if not torch.cuda.is_available():
        return None, None, None
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        device_used = (total_b - free_b) / (1024 ** 2)
    except Exception:
        device_used = None
    return allocated, reserved, device_used

# Discover test image paths from the dataset structure.
test_img_dir = ROOT / "ppe_sh17_yolo" / "images" / "test"
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
    image_paths.extend(test_img_dir.glob(ext))
image_paths = sorted(image_paths)
if MAX_IMAGES is not None:
    image_paths = image_paths[:MAX_IMAGES]
if not image_paths:
    raise FileNotFoundError(f"No test images found in {test_img_dir}")

print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"Test images: {len(image_paths)}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: CUDA unavailable; benchmarking on CPU")

# Warm-up.
first = cv2.imread(str(image_paths[0]))
if first is None:
    raise RuntimeError(f"Could not read image: {image_paths[0]}")
for _ in range(WARMUP_RUNS):
    _ = model.predict(first, imgsz=IMGSZ, device=DEVICE, verbose=False, conf=CONF)
cuda_sync()

# Sample RAM/VRAM in the background during benchmark.
samples = []
stop_sampler = threading.Event()

def sampler():
    while not stop_sampler.is_set():
        alloc, reserved, device_used = get_vram_mb()
        samples.append({
            "time": time.perf_counter(),
            "ram_mb": get_ram_mb(),
            "vram_allocated_mb": alloc,
            "vram_reserved_mb": reserved,
            "gpu_device_used_mb": device_used,
        })
        stop_sampler.wait(0.05)  # sample every 50 ms

sampler_thread = threading.Thread(target=sampler, daemon=True)
sampler_thread.start()

model_only_ms = []
end_to_end_ms = []
read_failures = 0
total_detections = 0

for idx, path in enumerate(image_paths, start=1):
    # End-to-end timer starts before image read and includes prediction and drawing.
    e2e_start = time.perf_counter()
    frame = cv2.imread(str(path))
    if frame is None:
        read_failures += 1
        continue

    cuda_sync()
    infer_start = time.perf_counter()
    result = model.predict(
        frame,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
        conf=CONF,
    )[0]
    cuda_sync()
    infer_end = time.perf_counter()
    model_only_ms.append((infer_end - infer_start) * 1000.0)

    total_detections += len(result.boxes) if result.boxes is not None else 0

    # Include annotation/drawing in the end-to-end processing time.
    annotated = result.plot()
    _ = annotated.shape
    e2e_end = time.perf_counter()
    end_to_end_ms.append((e2e_end - e2e_start) * 1000.0)

    if idx % 100 == 0:
        print(f"Processed {idx}/{len(image_paths)} images")

stop_sampler.set()
sampler_thread.join(timeout=2)

def summarize_ms(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(np.mean(a)),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "fps_from_mean": float(1000.0 / np.mean(a)) if np.mean(a) > 0 else 0.0,
    }

model_stats = summarize_ms(model_only_ms)
e2e_stats = summarize_ms(end_to_end_ms)

valid_samples = samples
ram_values = [s["ram_mb"] for s in valid_samples if s["ram_mb"] is not None]
vram_alloc_values = [s["vram_allocated_mb"] for s in valid_samples if s["vram_allocated_mb"] is not None]
vram_reserved_values = [s["vram_reserved_mb"] for s in valid_samples if s["vram_reserved_mb"] is not None]
gpu_used_values = [s["gpu_device_used_mb"] for s in valid_samples if s["gpu_device_used_mb"] is not None]

# Run standard Ultralytics validation to create confusion matrix and curves.
print("\nRunning test-set validation and creating standard plots...")
metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMGSZ,
    batch=4,
    workers=0,
    device=DEVICE,
    plots=True,
    project=str(OUT_DIR / "validation"),
    name="test",
    exist_ok=True,
)
val_dir = Path(metrics.save_dir)

# Per-class PR and F1 plots are also saved by Ultralytics in val_dir.
# Make a custom per-class grouped bar chart from returned metrics.
names = list(model.names.values())
try:
    precision = np.asarray(metrics.box.p) * 100
    recall = np.asarray(metrics.box.r) * 100
    ap50 = np.asarray(metrics.box.ap50) * 100
    ap = np.asarray(metrics.box.ap) * 100
    x = np.arange(len(names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - 1.5 * width, precision, width, label="Precision")
    ax.bar(x - 0.5 * width, recall, width, label="Recall")
    ax.bar(x + 0.5 * width, ap50, width, label="mAP@50")
    ax.bar(x + 1.5 * width, ap, width, label="mAP@50–95")
    ax.set_title("PPE Detection: Per-Class Test Metrics")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_metrics.png", dpi=180)
    plt.close(fig)
except Exception as exc:
    print(f"Could not create custom per-class metrics plot: {exc}")

# Plot loss curves from training results.csv if present.
candidate_csvs = [
    ROOT / r"runs\detect\runs\ppe_sh17\train\results.csv",
    ROOT / r"runs\ppe_sh17\train\results.csv",
]
# Also search the relevant training folder in case the run path differs.
candidate_csvs += list((ROOT / "runs").rglob("results.csv"))
results_csv = next((p for p in candidate_csvs if p.exists() and "ppe_sh17" in str(p).lower()), None)

if results_csv:
    print(f"Training history found: {results_csv}")
    with results_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []

    # Ultralytics versions commonly use these names. Only plot columns present.
    loss_columns = [
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
    ]
    available = [c for c in loss_columns if c in columns]
    if available and rows:
        epochs = np.arange(1, len(rows) + 1)
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in available:
            vals = []
            for row in rows:
                try:
                    vals.append(float(row[col]))
                except (ValueError, TypeError):
                    vals.append(np.nan)
            ax.plot(epochs, vals, label=col)
        ax.set_title("Training and Validation Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "loss_curves.png", dpi=180)
        plt.close(fig)
    else:
        print("No expected loss columns found in results.csv.")
        print("Available columns:", columns)
else:
    print("No SH17 results.csv found; cannot reconstruct training loss curves.")
    print("Find the training run's results.csv, then rerun or copy it into the project.")

# Export benchmark values to a CSV for the paper.
report = [
    ("Model checkpoint", str(MODEL_PATH)),
    ("Device", str(DEVICE)),
    ("GPU name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU run)"),
    ("Image size", f"{IMGSZ}x{IMGSZ}"),
    ("Batch size for inference benchmark", str(BATCH)),
    ("Images successfully processed", str(len(model_only_ms))),
    ("Image read failures", str(read_failures)),
    ("Total detections across processed images", str(total_detections)),
    ("Model-only mean latency (ms/frame)", f"{model_stats['mean_ms']:.3f}"),
    ("Model-only median latency (ms/frame)", f"{model_stats['median_ms']:.3f}"),
    ("Model-only P95 latency (ms/frame)", f"{model_stats['p95_ms']:.3f}"),
    ("Model-only FPS from mean latency", f"{model_stats['fps_from_mean']:.2f}"),
    ("End-to-end mean latency (ms/frame)", f"{e2e_stats['mean_ms']:.3f}"),
    ("End-to-end median latency (ms/frame)", f"{e2e_stats['median_ms']:.3f}"),
    ("End-to-end P95 latency (ms/frame)", f"{e2e_stats['p95_ms']:.3f}"),
    ("End-to-end FPS from mean latency", f"{e2e_stats['fps_from_mean']:.2f}"),
    ("Peak sampled process RAM (MB)", f"{max(ram_values):.1f}" if ram_values else "N/A"),
    ("Peak sampled PyTorch VRAM allocated (MB)", f"{max(vram_alloc_values):.1f}" if vram_alloc_values else "N/A"),
    ("Peak sampled PyTorch VRAM reserved (MB)", f"{max(vram_reserved_values):.1f}" if vram_reserved_values else "N/A"),
    ("Peak sampled GPU device memory used (MB)", f"{max(gpu_used_values):.1f}" if gpu_used_values else "N/A"),
    ("Validation plots folder", str(val_dir)),
    ("Training results.csv", str(results_csv) if results_csv else "Not found"),
]
with (OUT_DIR / "benchmark_report.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Measured value"])
    writer.writerows(report)

print("\n========== BENCHMARK SUMMARY ==========")
for label, value in report:
    print(f"{label}: {value}")
print(f"\nSaved report: {OUT_DIR / 'benchmark_report.csv'}")
print(f"Saved plots folder: {OUT_DIR}")
print("Look for Ultralytics PR_curve.png, F1_curve.png, confusion_matrix.png, and normalized confusion matrix in the validation folder.")
