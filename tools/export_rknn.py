# Add or modify code through Du Shenyu
#!/usr/bin/env python3
# YOLOv5 ONNX to RKNN export helper for Rockchip RK3566/RK3588.

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "runs/train/weld640_fgdc_vfm2/weights/best.onnx"
DEFAULT_OUT = ROOT / "runs/train/weld640_fgdc_vfm2/weights"


def parse_triplet(text):
    """Parse 'a,b,c' into [[a, b, c]] for RKNN config."""
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values, e.g. 0,0,0")
    return [values]


def check_dataset(dataset):
    if dataset is None:
        return None
    dataset = Path(dataset)
    if not dataset.is_file():
        raise FileNotFoundError(f"INT8 calibration dataset file not found: {dataset}")
    lines = [x.strip() for x in dataset.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines:
        raise ValueError(f"INT8 calibration dataset file is empty: {dataset}")
    return dataset


def make_dataset_txt(image_dir, output_txt, count, seed):
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p.resolve() for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No calibration images found under: {image_dir}")

    images = sorted(images)
    if count and len(images) > count:
        rng = random.Random(seed)
        images = sorted(rng.sample(images, count))

    output_txt = Path(output_txt)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    print(f"Calibration dataset written: {output_txt} ({len(images)} images)")
    return output_txt


def export_one(
    onnx_path,
    output_path,
    platform,
    quantized,
    dataset,
    mean_values,
    std_values,
    quantized_dtype,
    optimization_level,
):
    # Import lazily so this script can show --help on machines without RKNN-Toolkit2.
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ImportError(
            "RKNN-Toolkit2 is not installed in this Python environment. "
            "Install the Rockchip rknn-toolkit2 package on the conversion host first."
        ) from exc

    rknn = RKNN(verbose=True)
    try:
        print(f"\n=== RKNN export: platform={platform}, quantized={quantized} ===")
        print(f"ONNX: {onnx_path}")
        print(f"RKNN: {output_path}")

        ret = rknn.config(
            target_platform=platform,
            mean_values=[mean_values],
            std_values=[std_values],
            quantized_dtype=quantized_dtype,
            optimization_level=optimization_level,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")

        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")

        ret = rknn.build(do_quantization=quantized, dataset=str(dataset) if quantized else None)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")

        print(f"Exported: {output_path}")
    finally:
        rknn.release()


def build_outputs(args):
    modes = ["fp16", "int8"] if args.mode == "both" else [args.mode]
    outputs = []
    for platform in args.platforms:
        for mode in modes:
            quantized = mode == "int8"
            suffix = "int8" if quantized else "fp16"
            output_name = f"{args.name}_{platform}_{suffix}.rknn"
            outputs.append((platform, mode, quantized, args.output_dir / output_name))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLOv5 ONNX model to RKNN for RK3566/RK3588.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path to exported ONNX model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help="Directory for .rknn files.")
    parser.add_argument("--name", default="best", help="Output filename prefix.")
    parser.add_argument(
        "--platforms",
        nargs="+",
        default=["rk3566", "rk3588"],
        choices=["rk3566", "rk3568", "rk3588"],
        help="Rockchip target platforms to export.",
    )
    parser.add_argument(
        "--mode",
        default="fp16",
        choices=["fp16", "int8", "both"],
        help="Export precision. Use int8/both only with a calibration dataset.",
    )
    parser.add_argument("--dataset", type=Path, help="RKNN INT8 calibration dataset txt file.")
    parser.add_argument(
        "--make-dataset-from",
        type=Path,
        help="Image directory used to generate calibration dataset txt before INT8 export.",
    )
    parser.add_argument("--dataset-count", type=int, default=300, help="Max calibration images to sample.")
    parser.add_argument("--dataset-seed", type=int, default=0, help="Random seed for calibration image sampling.")
    parser.add_argument(
        "--mean-values",
        type=parse_triplet,
        default=[[0.0, 0.0, 0.0]],
        help="Input mean values as R,G,B. Default: 0,0,0",
    )
    parser.add_argument(
        "--std-values",
        type=parse_triplet,
        default=[[255.0, 255.0, 255.0]],
        help="Input std values as R,G,B. Default: 255,255,255",
    )
    parser.add_argument(
        "--quantized-dtype",
        default="asymmetric_quantized-8",
        help="RKNN quantized dtype for INT8 builds.",
    )
    parser.add_argument("--optimization-level", type=int, default=3, help="RKNN optimization level.")
    return parser.parse_args()


def main(args):
    args.onnx = args.onnx.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    dataset = None
    if args.mode in {"int8", "both"}:
        if args.dataset is None and args.make_dataset_from is not None:
            args.dataset = args.output_dir / f"{args.name}_calibration_dataset.txt"
            make_dataset_txt(args.make_dataset_from, args.dataset, args.dataset_count, args.dataset_seed)
        dataset = check_dataset(args.dataset)
    mean_values = args.mean_values[0]
    std_values = args.std_values[0]

    for platform, mode, quantized, output_path in build_outputs(args):
        if quantized and dataset is None:
            raise ValueError("INT8 export requires --dataset calibration txt.")
        export_one(
            onnx_path=args.onnx,
            output_path=output_path,
            platform=platform,
            quantized=quantized,
            dataset=dataset,
            mean_values=mean_values,
            std_values=std_values,
            quantized_dtype=args.quantized_dtype,
            optimization_level=args.optimization_level,
        )


if __name__ == "__main__":
    main(parse_args())
