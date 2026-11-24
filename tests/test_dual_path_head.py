# Add or modify code through Du Shenyu
import torch
import yaml

from models.yolo import Model


def _dual_path_cfg(nc=5, use_fgdh=False):
    with open("models/yolov5n.yaml", encoding="ascii", errors="ignore") as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = nc
    cfg["head_type"] = "dual_path"
    if use_fgdh:
        cfg["head"][-1][-1].append({"use_fgdh": True, "fgdh_compact_dim": 64})
    return cfg


def test_dual_path_detect_forward_backward_shape():
    nc = 5
    model = Model(_dual_path_cfg(nc), ch=3, nc=nc)
    model.train()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    assert type(model.model[-1]).__name__ == "DualPathDetect"
    assert [tuple(t.shape) for t in y] == [
        (2, 3, 28, 28, 5 + nc),
        (2, 3, 14, 14, 5 + nc),
        (2, 3, 7, 7, 5 + nc),
    ]

    loss = sum(t.sum() for t in y)
    loss.backward()
    assert any(p.grad is not None for p in model.model[-1].parameters())


def test_dual_path_detect_inference_format():
    nc = 5
    model = Model(_dual_path_cfg(nc), ch=3, nc=nc)
    model.eval()

    with torch.no_grad():
        pred, train_out = model(torch.randn(2, 3, 224, 224))

    assert tuple(pred.shape) == (2, 3087, 5 + nc)
    assert [tuple(t.shape) for t in train_out] == [
        (2, 3, 28, 28, 5 + nc),
        (2, 3, 14, 14, 5 + nc),
        (2, 3, 7, 7, 5 + nc),
    ]


def test_dual_path_detect_with_fgdh_forward_backward_shape():
    nc = 5
    model = Model(_dual_path_cfg(nc, use_fgdh=True), ch=3, nc=nc)
    model.train()

    y = model(torch.randn(2, 3, 224, 224))

    assert type(model.model[-1].cls_stem[0]).__name__ == "FGDH"
    assert [tuple(t.shape) for t in y] == [
        (2, 3, 28, 28, 5 + nc),
        (2, 3, 14, 14, 5 + nc),
        (2, 3, 7, 7, 5 + nc),
    ]

    loss = sum(t.sum() for t in y)
    loss.backward()
    assert any(p.grad is not None for p in model.model[-1].cls_stem.parameters())
