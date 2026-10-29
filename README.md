Here is the complete list of software, datasets, libraries, and files used or prepared for your Industrial PPE Detection project, without verification commands.

## 1. Software and Python libraries

| Name                                 | Purpose                                        |
| ------------------------------------ | ---------------------------------------------- |
| Python                               | Runs the project code                          |
| Python virtual environment (`.venv`) | Isolates project dependencies                  |
| Ultralytics YOLO                     | Object detection, training and validation      |
| PyTorch                              | Deep-learning framework                        |
| OpenCV (`opencv-python`)             | Webcam access, bounding boxes and live display |
| NumPy                                | Numerical operations                           |
| Matplotlib                           | Graphs and evaluation plots                    |
| psutil                               | RAM usage measurement for benchmarking         |
| Command Prompt (CMD)                 | Runs Python scripts                            |

## 2. Datasets downloaded

| Dataset          | Details                                                        |
| ---------------- | -------------------------------------------------------------- |
| Construction-PPE | Original dataset used for initial model training               |
| SH17             | 8,099 images, 75,994 annotated objects and 17 original classes |

SH17 was prepared into five classes: person, gloves, helmet, shoes and safety vest.

- Original SH17 archive: `C:\Users\ladhr\Downloads\archive`
- Prepared dataset: `ppe_sh17_yolo`
- Training set: 6,479 images
- Validation set: 810 images
- Test set: 810 images

Source: [SH17 official repository](https://github.com/ahmadmughees/SH17dataset?utm_source=chatgpt.com)

## 3. Model weights

- `yolo11n.pt` — pretrained YOLO checkpoint used during initial setup.
- `runs/detect/runs/ppe/train/weights/best.pt` — original model used by your current webcam script.
- `runs/ppe_improved/train/weights/best.pt` — improved-model experiment.
- `runs/detect/runs/ppe_sh17/train/weights/best.pt` — SH17-trained model.
- `last.pt` — last-epoch checkpoint, if retained.

## 4. Python scripts and configuration files

- `train.py` — initial model training.
- `evaluate.py` — initial model evaluation.
- `detect_live.py` — real-time webcam detection with confidence percentages.
- `prepare_construction_ppe.py` — dataset preparation.
- `prepare_ppe_dataset.py` — SH17 class selection and dataset splitting.
- `train_sh17.py` — SH17 model training.
- `evaluate_sh17.py` — test-set evaluation.
- `evaluate_sh17_graphs.py` — evaluation graphs and confusion matrix.
- `benchmark_ppe.py` — FPS, latency, memory and evaluation benchmarking.
- `data.yaml` — dataset paths and class definitions.
- `results.csv` — training history used for loss curves, if retained.
- `requirements.txt` — package-version list, if generated.

## 5. Generated outputs

- `best.pt` — best-performing saved checkpoint for a training run.
- `last.pt` — final-epoch checkpoint, if retained.
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `PR_curve.png`
- `F1_curve.png`
- `P_curve.png`
- `R_curve.png`
- `per_class_metrics.png`
- `loss_curves.png`
- `benchmark_report.csv`

The graph and report files are generated outputs, not downloaded software.