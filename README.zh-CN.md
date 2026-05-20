# FGDC-Net

**Fine-Grained Industrial Defect Detection via Dual-Path Feature Decoupling**

FGDC-Net 是一个面向工业缺陷检测的 YOLO 系列改进模型，主要用于小目标、弱纹理、类别相似、边界模糊等细粒度缺陷场景。

本项目包含 YOLOv5 版本，也提供了可适配 Ultralytics YOLOv5、YOLOv8、YOLO11 风格配置的实现。

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-supported-red)
![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-green)
![ONNX](https://img.shields.io/badge/ONNX-supported-orange)
![RKNN](https://img.shields.io/badge/RKNN-RK3566%20%7C%20RK3588-purple)

> **说明**
> 本项目不包含预训练 teacher 权重和工业数据集，请使用自己的数据与 teacher checkpoint 进行复现实验。

## 特性

- **Dual-Path Detection Head**：在检测头中解耦分类特征和定位特征。
- **FGDH**：通过双卷积分支和紧凑双线性融合增强细粒度分类能力。
- **VFM Feature Guider**：训练阶段使用分类 teacher 和回归 teacher 进行特征指导。
- **Branch-Specific VFM Loss**：支持分类语义关系蒸馏和前景加权回归蒸馏。
- **Ultralytics 适配**：支持 YOLOv5、YOLOv8、YOLO11 风格配置。
- **部署友好**：teacher 只在训练阶段使用，导出和推理模型不包含 teacher 编码器。

## 模型配置

YOLOv5 配置：

```text
models/FGDCn-dualpath.yaml
models/FGDCn-fgdc-vfm.yaml
models/FGDCn-fgdc-vfm-placeholder.yaml
models/FGDCn-fgdc-vfm-branchloss.yaml
```

Ultralytics 配置：

```text
ultralytics-main/ultralytics/cfg/models/fgdc/yolov5-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolov8-fgdc-vfm.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm.yaml
ultralytics-main/ultralytics/cfg/models/fgdc/yolo11-fgdc-vfm-branchloss.yaml
```

VFM teacher 仅用于训练。导出的 ONNX 和 RKNN 模型只保留轻量检测器。

## 环境安装

```bash
git clone your_repo_url
cd FGDCNet
pip install -r requirements.txt
pip install timm onnx onnxruntime onnxsim pytest
```

Ultralytics 版本：

```bash
cd ultralytics-main
pip install -e .
```

## 数据集格式

使用标准 YOLO 数据格式：

```text
your_dataset/
  train/images/
  train/labels/
  val/images/
  val/labels/
```

数据集 yaml 示例：

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

## 训练

YOLOv5 FGDC-Net，使用 Branch-Specific VFM Loss：

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

只使用 Dual-Path + FGDH：

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

Ultralytics YOLO11 FGDC-Net：

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

## VFM 损失

普通 MSE 蒸馏：

```yaml
VFM_cls_loss: MSE
VFM_reg_loss: MSE
```

分支特定蒸馏：

```yaml
use_vfm_guider: true
VFM_cls_loss: BranchSpecific
VFM_reg_loss: BranchSpecific
vfm_alpha: 0.5
vfm_beta: 0.5
vfm_max_tokens: 256
```

实现位置：

```text
losses/vfm_guidance_loss.py
```

训练日志会记录：

```text
loss_cls_cos
loss_cls_rel
loss_reg_fg
loss_reg_att
loss_vfm
```

## 导出

ONNX：

```bash
python export.py \
  --weights runs/train/your_exp/weights/best.pt \
  --include onnx \
  --img 640 \
  --batch-size 1 \
  --device cpu \
  --opset 18
```

ONNX 简化：

```bash
python -m onnxsim \
  runs/train/your_exp/weights/best.onnx \
  runs/train/your_exp/weights/best_sim.onnx
```

RKNN FP16：

```bash
python tools/export_rknn.py \
  --onnx runs/train/your_exp/weights/best_sim.onnx \
  --platforms rk3566 rk3588 \
  --mode fp16 \
  --name your_fgdc_model \
  --output-dir runs/train/your_exp/weights
```

RKNN INT8：

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

## 注意事项

- teacher 编码器只在训练阶段使用。
- 推理、ONNX 导出、RKNN 导出不会加载 DINO、ViT 或 Swin teacher。
- `best.pt` 可能包含训练阶段状态，体积可能大于导出的 ONNX/RKNN 模型。
- RKNN 部署建议先测试 FP16，再使用代表性校准集评估 INT8。

## 致谢

本项目基于 PyTorch、YOLOv5、Ultralytics YOLO、ONNX、RKNN Toolkit 和 timm 构建。
