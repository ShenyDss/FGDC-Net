# 🧠 FGDC-Net

<div align="center">

## Fine-Grained Industrial Defect Detection via Dual-Path Feature Decoupling

**A YOLO-based fine-grained industrial defect detector for robust industrial defect inspection.**

<br>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-green)
![ONNX](https://img.shields.io/badge/ONNX-Export-supported-orange)
![RKNN](https://img.shields.io/badge/RKNN-RK3566%20%7C%20RK3588-purple)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey)

</div>

---

## 📌 Overview

**FGDC-Net** is a fine-grained industrial defect detection framework built upon the YOLO detection family.

It is designed for challenging industrial inspection scenarios where defects are often:

- small in scale,
- visually similar,
- weak in texture,
- irregular in shape,
- and difficult to distinguish using standard detection heads.

To address these challenges, FGDC-Net introduces a **dual-path feature decoupling framework**, which separates classification-oriented and localization-oriented features before prediction.

The framework mainly contains:

- 🔀 **Dual-path decoupled detection head**
- 🔬 **Fine-Grained Enhanced Classification Head, FGDH**
- 🧩 **VFM-guided feature learning strategy**
- 🚀 **Clean deployment-friendly inference model**

> **Note**  
> Due to privacy restrictions and potential conflicts of interest, the pre-trained weights and industrial datasets used in this study are not open-sourced.  
> Researchers may use this repository to conduct replication experiments on their own industrial datasets.

---

## ✨ Highlights

| Feature | Description |
|---|---|
| 🔀 **Dual-Path Head** | Separates classification-oriented and localization-oriented features to reduce task conflict. |
| 🔬 **FGDH** | Enhances fine-grained classification with dual convolution streams and compact bilinear fusion. |
| 🧩 **VFM Guider** | Uses ViT/DINO-style and Swin teacher encoders for training-time feature guidance. |
| ⚙️ **Ultralytics Support** | Provides YOLOv5, YOLOv8, and YOLO11-style model configurations. |
| 🚀 **Edge Deployment** | Supports ONNX export and RKNN deployment on RK3566 / RK3588 platforms. |
| 🧼 **Clean Inference Graph** | Teacher models are used only during training and removed during deployment. |

---

## 🏗️ Framework

```text
Input Image
    │
    ▼
YOLO Backbone + Neck
    │
    ▼
Dual-Path Feature Decoupling
    │
    ├── Classification Path
    │       │
    │       ▼
    │     FGDH
    │       ├── Local Texture Stream
    │       ├── Context Enhancement Stream
    │       └── Compact Bilinear Fusion
    │
    └── Localization Path
            │
            ▼
        Box Regression Branch

Training Stage:
    └── Optional VFM Feature Guider
        ├── Classification Teacher
        └── Localization Teacher

Inference Stage:
    └── Clean YOLO-compatible detector
        without ViT / DINO / Swin teacher encoders
```

---

## 📁 Repository Structure

```text
FGDCNet/
├── models/
│   ├── FGDCn-dualpath.yaml
│   ├── FGDCn-fgdc-vfm.yaml
│   └── ...
│
├── ultralytics-main/
│   └── ultralytics/
│       └── cfg/
│           └── models/
│               └── fgdc/
│                   ├── yolov5-fgdc.yaml
│                   ├── yolov8-fgdc.yaml
│                   ├── yolov8-fgdc-vfm.yaml
│                   ├── yolo11-fgdc.yaml
│                   └── yolo11-fgdc-vfm.yaml
│
├── tools/
│   └── export_rknn.py
│
├── train.py
├── export.py
├── requirements.txt
└── README.md
```

---

## 📦 Supported Configurations

### FGDCn Configs

```text
models/FGDCn-dualpath.yaml
models/FGDCn-fgdc-vfm.yaml
```

### Ultralytics Configs

```text
ultralytics-main/ultralytics/cfg/models/fgdc/yolov5-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc-vfm.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm.yaml
```

The Ultralytics implementation uses `FGDCDetect` as a Detect-compatible head.  
Therefore, official model configuration styles and normal training behavior are preserved.

---

## 🔧 Installation

### Basic Environment

```bash
cd path/to/FGDCNet

pip install -r requirements.txt
pip install timm onnx onnxscript onnxruntime onnxsim pytest
```

### Ultralytics Version

```bash
cd path/to/FGDCNet/ultralytics-main

pip install -e .
```

---

## 📦 Teacher Weights

Teacher models are used only during training.

Example teacher weight paths:

```text
path/to/pretrain/your_cls_teacher.pt
path/to/pretrain/your_reg_teacher.pth
```

> **Deployment Note**  
> Exported ONNX and RKNN models do **not** contain ViT, DINO, Swin, or other teacher encoders.  
> The inference model remains clean and deployment-friendly.

---

## 🗂️ Dataset Format

FGDC-Net follows the standard YOLO dataset format.

```text
your_dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
```

Example dataset YAML:

```yaml
path: path/to/your_dataset

train: train/images
val: val/images

names:
  0: your_class_0
  1: your_class_1
```

For example, if your dataset contains five defect categories:

```yaml
path: datasets/your_industrial_dataset

train: train/images
val: val/images

names:
  0: defect_0
  1: defect_1
  2: defect_2
  3: defect_3
  4: defect_4
```

---

## 🚀 Training

### FGDCn Full FGDC-Net

```powershell
python train.py `
  --img 640 `
  --batch 4 `
  --epochs 100 `
  --data data/your_dataset.yaml `
  --cfg models/FGDCn-fgdc-vfm.yaml `
  --weights "" `
  --name your_fgdc_exp `
  --vfm-cls-weights path/to/pretrain/your_cls_teacher.pt `
  --vfm-reg-weights path/to/pretrain/your_reg_teacher.pth `
  --vfm-imgsz 224
```

---

### Ultralytics YOLO11 FGDC-Net

```bash
cd path/to/FGDCNet/ultralytics-main

yolo detect train \
  model=ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm.yaml \
  data=data/your_dataset.yaml \
  imgsz=640 \
  batch=4 \
  epochs=100
```

---

### Ultralytics YOLOv8 FGDC-Net

```bash
cd path/to/FGDCNet/ultralytics-main

yolo detect train \
  model=ultralytics/cfg/models/fgdc/yolov8-fgdc.yaml \
  data=data/your_dataset.yaml \
  imgsz=640 \
  batch=4 \
  epochs=100
```

---

### Ultralytics YOLOv5 FGDC-Net

```bash
cd path/to/FGDCNet/ultralytics-main

yolo detect train \
  model=ultralytics/cfg/models/fgdc/yolov5-fgdc.yaml \
  data=data/your_dataset.yaml \
  imgsz=640 \
  batch=4 \
  epochs=100
```

---

## 🧪 Model Build Check

You can verify whether the model is correctly built with `FGDCDetect`.

```bash
python -c "from ultralytics.nn.tasks import DetectionModel; m=DetectionModel('ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm.yaml', ch=3, nc=5, verbose=True); print(type(m.model[-1]).__name__)"
```

Expected output:

```text
FGDCDetect
```

---

## 📤 Export

### ONNX Export

```powershell
python export.py `
  --weights runs/train/your_exp/weights/best.pt `
  --include onnx `
  --img 640 `
  --batch-size 1 `
  --device cpu `
  --opset 18
```

### ONNX Simplification

```powershell
python -m onnxsim `
  runs/train/your_exp/weights/best.onnx `
  runs/train/your_exp/weights/best_sim.onnx
```

---

## 💻 RKNN Deployment

### FP16 Export for RK3566 / RK3588

```bash
python tools/export_rknn.py \
  --onnx runs/train/your_exp/weights/best_sim.onnx \
  --platforms rk3566 rk3588 \
  --mode fp16 \
  --name your_fgdc_model \
  --output-dir runs/train/your_exp/weights
```

### INT8 Export with Calibration Dataset

```bash
python tools/export_rknn.py \
  --onnx runs/train/your_exp/weights/best_sim.onnx \
  --platforms rk3566 rk3588 \
  --mode int8 \
  --make-dataset-from path/to/your_dataset/train/images \
  --dataset-count 300 \
  --name your_fgdc_model \
  --output-dir runs/train/your_exp/weights
```

---

## 📊 Recommended Deployment Strategy

| Stage | Recommendation |
|---|---|
| Step 1 | Export the trained model to ONNX. |
| Step 2 | Simplify the ONNX model using `onnxsim`. |
| Step 3 | Start RKNN deployment with FP16. |
| Step 4 | Compare INT8 speed and accuracy. |
| Step 5 | Use INT8 only when the accuracy drop is acceptable. |

---

## 🧩 Training and Inference Behavior

| Component | Training | Inference / Export |
|---|---:|---:|
| YOLO Backbone | ✅ | ✅ |
| YOLO Neck | ✅ | ✅ |
| Dual-Path Head | ✅ | ✅ |
| FGDH | ✅ | ✅ |
| VFM Classification Teacher | ✅ | ❌ |
| VFM Localization Teacher | ✅ | ❌ |
| ViT / DINO / Swin Encoders | ✅ | ❌ |
| ONNX Export Graph | — | ✅ Clean |
| RKNN Export Graph | — | ✅ Clean |

---

## ⚠️ Important Notes

- Teacher models are used **only during training**.
- `best.pt` may be large if teacher modules are saved in the checkpoint.
- `best.onnx`, `best_sim.onnx`, and `.rknn` files do **not** contain teacher encoders.
- Please ensure that all `devices` are consistent during training and export.
- For deployment, FP16 RKNN is recommended as the first baseline before INT8 quantization.
- INT8 export requires representative calibration images.
- The Ultralytics version keeps `FGDCDetect` compatible with normal YOLO detection behavior.

---

## ❓ FAQ

### 1. Are teacher models required during inference?

No. Teacher models are used only during training for feature guidance.  
The exported ONNX and RKNN models are clean inference models.

### 2. Why is `best.pt` sometimes large?

The training checkpoint may contain teacher-related modules or additional training-time states.  
However, the exported ONNX / RKNN models do not contain teacher encoders.

### 3. Can I use FGDC-Net on my own industrial dataset?

Yes. You can use the standard YOLO dataset format and modify the dataset YAML file according to your own categories.

### 4. Does FGDC-Net support Ultralytics YOLO?

Yes. This repository provides YOLOv5, YOLOv8, and YOLO11-style configurations under:

```text
ultralytics-main/ultralytics/cfg/models/fgdc/
```

### 5. Is the inference speed affected by VFM teachers?

No. VFM teachers are removed during inference and export.  
Only the clean detector is deployed.

---

---

## 🙏 Acknowledgements

This project is built upon the YOLO detection ecosystem and benefits from the open-source contributions of:

- YOLOv5
- Ultralytics YOLOv8 / YOLO11
- PyTorch
- ONNX
- RKNN Toolkit
- timm

---

<div align="center">

## ⭐ FGDC-Net

**Fine-grained defect detection for industrial inspection and edge deployment.**

</div>
