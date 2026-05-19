# Add or modify code through Du Shenyu
"""Inspect and optionally load FGDC-Net VFM teacher checkpoints.

Usage:
    python tools/inspect_vfm_weights.py --cls-weights D:/FGDCNet/pre-train/vit_base_teacher_best.pt --reg-weights D:/FGDCNet/pre-train/Swin_base_model_20ep.pth

If timm is installed, this script also instantiates matching teacher models,
loads the state dicts, and prints candidate feature output shapes.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.detectron2_swin import SwinTransformer


def load_state_dict(path):
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "teacher", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return ckpt


def summarize_state_dict(name, state_dict):
    keys = list(state_dict.keys())
    print(f"\n=== {name} ===")
    print(f"num_keys: {len(keys)}")
    print(f"first_keys: {keys[:12]}")
    print(f"last_keys: {keys[-12:]}")

    tensor_shapes = {k: tuple(v.shape) for k, v in state_dict.items() if torch.is_tensor(v)}
    if "patch_embed.proj.weight" in tensor_shapes:
        print(f"patch_embed.proj.weight: {tensor_shapes['patch_embed.proj.weight']}")
    if "pos_embed" in tensor_shapes:
        print(f"pos_embed: {tensor_shapes['pos_embed']}")
    print(f"head_keys: {[k for k in keys if k.startswith('head')][:8]}")

    if any(k.startswith("blocks.") for k in keys):
        print("detected_arch: ViT base patch16 style")
        print("recommended_feature: final patch tokens after norm, drop cls token, reshape to Bx768x14x14 for 224 input")
    if any(k.startswith("layers.") for k in keys):
        stage_norms = [k for k in keys if k.startswith("norm") and k.endswith(".weight")]
        print("detected_arch: Swin base patch4 window7 style")
        print(f"stage_norms: {stage_norms}")
        print("recommended_feature: final stage feature, or multi-scale stage features; first use final Bx1024x7x7 for 224 input")


def try_timm_load(cls_sd, reg_sd, img_size):
    try:
        import timm
    except Exception as e:
        print(f"\ntimm_unavailable: {type(e).__name__}: {e}")
        print("Install timm to run model-level load/forward checks.")
        return

    print(f"\ntimm_version: {timm.__version__}")

    cls_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    cls_msg = cls_model.load_state_dict(cls_sd, strict=False)
    print("\nvit_base_patch16_224 load_state_dict(strict=False):")
    print(f"missing_keys: {cls_msg.missing_keys[:20]}")
    print(f"unexpected_keys: {cls_msg.unexpected_keys[:20]}")

    reg_model = SwinTransformer(
        pretrain_img_size=img_size,
        patch_size=4,
        in_chans=3,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        out_indices=(0, 1, 2, 3),
    )
    reg_msg = reg_model.load_state_dict(reg_sd, strict=True)
    print("\ndetectron2-style SwinTransformer load_state_dict(strict=True):")
    print(f"missing_keys: {reg_msg.missing_keys}")
    print(f"unexpected_keys: {reg_msg.unexpected_keys}")

    cls_model.eval()
    reg_model.eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        cls_tokens = cls_model.forward_features(x)
        reg_feats = reg_model(x)
    print(f"\nvit forward_features output: {tuple(cls_tokens.shape)}")
    if cls_tokens.ndim == 3 and cls_tokens.shape[1] == (img_size // 16) ** 2 + 1:
        print(f"vit patch-token feature: {(1, cls_tokens.shape[-1], img_size // 16, img_size // 16)}")
    print(f"swin outputs: { {k: tuple(v.shape) for k, v in reg_feats.items()} }")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cls-weights", type=str, default=r"D:\FGDCNet\pre-train\vit_base_teacher_best.pt")
    parser.add_argument("--reg-weights", type=str, default=r"D:\FGDCNet\pre-train\Swin_base_model_20ep.pth")
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def main(opt):
    cls_sd = load_state_dict(opt.cls_weights)
    reg_sd = load_state_dict(opt.reg_weights)
    summarize_state_dict(Path(opt.cls_weights).name, cls_sd)
    summarize_state_dict(Path(opt.reg_weights).name, reg_sd)
    try_timm_load(cls_sd, reg_sd, opt.img_size)


if __name__ == "__main__":
    main(parse_opt())
