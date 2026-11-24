# FGDC-Net

**Fine-Grained Industrial Defect Detection via Dual-Path Feature Decoupling**

FGDC-Net is a YOLO-based detector for fine-grained industrial defect inspection. It targets small, low-contrast, visually similar, and boundary-ambiguous defects that are hard for a standard coupled detection head.

This repository includes a YOLOv5 implementation and an Ultralytics-compatible implementation for YOLOv5, YOLOv8, and YOLO11 style models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)
![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-green)
![ONNX](https://img.shields.io/badge/ONNX-supported-orange)
![RKNN](https://img.shields.io/badge/RKNN-RK3566%20%7C%20RK3588-purple)

> **Note**
> Pre-trained teacher weights and industrial datasets are not included. Use your own data and teacher checkpoints for reproduction or further research.

## Highlights

- **Dual-Path Detection Head**: decouples classification-oriented and localization-oriented features.
- **FGCH**: improves fine-grained classification with dual convolution streams and compact bilinear fusion.
- **VFM Feature Guider**: uses classification and regression teacher encoders during training only.
- **Branch-Specific VFM Loss**: adds semantic relation guidance and foreground-aware regression guidance.
- **Ultralytics Support**: works with YOLOv5, YOLOv8, and YOLO11 style configs.
- **Clean Export**: teacher encoders are not used during inference, ONNX export, or RKNN deployment.

## Model Variants

YOLOv5 configs:

```text
models/FGDCn-dualpath.yaml
models/FGDCn-fgdc-vfm.yaml
models/FGDCn-fgdc-vfm-placeholder.yaml
models/FGDCn-fgdc-vfm-branchloss.yaml
```

Ultralytics configs:

```text
ultralytics-main/ultralytics/cfg/models/fgdc/yolov5-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc-vfm.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm-branchloss.yaml
```

The VFM teachers are training-time helpers. Exported ONNX and RKNN models keep only the lightweight detector.

## Installation

```bash
git clone your_repo_url
cd FGDCNet
pip install -r requirements.txt
pip install timm onnx onnxruntime onnxsim pytest
```

For the Ultralytics version:

```bash
cd ultralytics-main
pip install -e .
```

## Dataset Format

Use the standard YOLO format:

```text
your_dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
```

Example dataset yaml:

```yaml
path: datasets/your_dataset
train: train/images
val: val/images

names:
  0: defect_0
  1: defect_1
  2: defect_2
  3: defect_3
  4: defect_4
```

## Training

YOLOv5 FGDC-Net with branch-specific VFM loss:

```bash
python train.py \
  --img 640 \
  --batch 4 \
  --epochs 100 \
  --data data/your_dataset.yaml \
  --cfg models/FGDCn-fgdc-vfm-branchloss.yaml \
  --weights "" \
  --name your_fgdc_exp \
  --vfm-cls-weights pre-train/your_cls_teacher.pt \
  --vfm-reg-weights pre-train/your_reg_teacher.pth \
  --vfm-imgsz 224
```

Dual-Path + FGCH only:

```bash
python train.py \
  --img 640 \
  --batch 4 \
  --epochs 100 \
  --data data/your_dataset.yaml \
  --cfg models/FGDCn-dualpath.yaml \
  --weights "" \
  --name your_dualpath_exp
```

Ultralytics YOLO11 FGDC-Net:

```bash
cd ultralytics-main

yolo detect train \
  model=ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm-branchloss.yaml \
  data=../data/your_dataset.yaml \
  imgsz=640 \
  batch=4 \
  epochs=100 \
  name=your_fgdc_yolo11_exp
```

## VFM Guidance

MSE guidance:

```yaml
VFM_cls_loss: MSE
VFM_reg_loss: MSE
```

Branch-specific guidance:

```yaml
use_vfm_guider: true
VFM_cls_loss: BranchSpecific
VFM_reg_loss: BranchSpecific
vfm_alpha: 0.5
vfm_beta: 0.5
vfm_max_tokens: 256
```

Implementation:

```text
losses/vfm_guidance_loss.py
```

Training logs include:

```text
loss_cls_cos
loss_cls_rel
loss_reg_fg
loss_reg_att
loss_vfm
```

## Export

ONNX:

```bash
python export.py \
  --weights runs/train/your_exp/weights/best.pt \
  --include onnx \
  --img 640 \
  --batch-size 1 \
  --device cpu \
  --opset 18
```

Simplify ONNX:

```bash
python -m onnxsim \
  runs/train/your_exp/weights/best.onnx \
  runs/train/your_exp/weights/best_sim.onnx
```

RKNN FP16:

```bash
python tools/export_rknn.py \
  --onnx runs/train/your_exp/weights/best_sim.onnx \
  --platforms rk3566 rk3588 \
  --mode fp16 \
  --name your_fgdc_model \
  --output-dir runs/train/your_exp/weights
```

RKNN INT8:

```bash
python tools/export_rknn.py \
  --onnx runs/train/your_exp/weights/best_sim.onnx \
  --platforms rk3566 rk3588 \
  --mode int8 \
  --make-dataset-from datasets/your_dataset/train/images \
  --dataset-count 300 \
  --name your_fgdc_model \
  --output-dir runs/train/your_exp/weights
```

## Notes

- Teacher encoders are used only during training.
- Inference, ONNX export, and RKNN export do not load DINO, ViT, or Swin teachers.
- `best.pt` can be larger than the exported model because it may include training-time states.
- Start RKNN deployment with FP16, then evaluate INT8 with a representative calibration set.

## Acknowledgments

This project builds on PyTorch, YOLOv5, Ultralytics YOLO, ONNX, RKNN Toolkit, and timm.
