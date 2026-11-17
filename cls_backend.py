"""Classifier backends for infer.py: each loader returns run(x) mapping an (N, 3, S, S) float32 array to (N, C) outputs.

  .onnx -> onnxruntime (the exported model the delivered app uses)
  .pth  -> the mmpretrain training checkpoint (SwinV2-base 384), run directly in PyTorch
  .pt   -> best.pt of mosquito_misslab_training (Swin V2 tiny / small / base species classifier); carries its own
           class names, mean/std, image size and temperature, and outputs sigmoid probabilities
"""
import numpy as np

# Architecture of the training config stored in the checkpoints (SwinV2-base, 384 px, 4 classes).
_BACKBONE = dict(type="SwinTransformerV2", arch="base", img_size=384, window_size=[24, 24, 24, 12],
                 pretrained_window_sizes=[24, 24, 24, 12], pad_small_map=True)
# Buffers recomputed in __init__ rather than saved in the checkpoint.
_DERIVED = ("relative_coords_table", "relative_position_index")


def load_onnx(path, providers, preload_dlls=True):
    import onnxruntime as ort
    if preload_dlls:
        try:
            ort.preload_dlls()  # find pip-installed CUDA/cuDNN libs
        except Exception:
            pass
    sess = ort.InferenceSession(str(path), providers=list(providers))
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: np.ascontiguousarray(x)})[0]


def load_pth(path, device=None):
    import torch
    from mmpretrain.models import ImageClassifier
    ck = torch.load(path, map_location="cpu", weights_only=False)  # pickles mmengine objects; trusted file only
    sd = ck.get("state_dict", ck)
    num_classes = sd["head.fc.weight"].shape[0]
    model = ImageClassifier(backbone=_BACKBONE, neck=dict(type="GlobalAveragePooling"),
                            head=dict(type="LinearClsHead", num_classes=num_classes, in_channels=sd["head.fc.weight"].shape[1]))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.endswith(_DERIVED)] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint {path} does not match the SwinV2-base architecture: {bad[:5]}")
    dev = f"cuda:{device}" if str(device).isdigit() else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    @torch.inference_mode()
    def run(x):
        return model(torch.from_numpy(np.ascontiguousarray(x)).to(dev), mode="tensor").float().cpu().numpy()
    return run


def load_species(ck, device=None):
    """Swin V2-T species classifier from a best.pt dict. run(x) -> sigmoid(logits / temperature), (N, n_classes)."""
    import torch
    from torchvision.models import swin_v2_b, swin_v2_s, swin_v2_t
    model = {"swin_v2_t": swin_v2_t, "swin_v2_s": swin_v2_s, "swin_v2_b": swin_v2_b}[ck["arch"]](weights=None)
    model.head = torch.nn.Linear(model.head.in_features, len(ck["classes"]))
    model.load_state_dict(ck["model"])
    dev = f"cuda:{device}" if str(device).isdigit() else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    @torch.inference_mode()
    def run(x):
        with torch.autocast(torch.device(dev).type, dtype=torch.bfloat16, enabled=dev != "cpu"):
            out = model(torch.from_numpy(np.ascontiguousarray(x)).to(dev))
        return (out.float() / ck["temperature"]).sigmoid().cpu().numpy()
    run.is_prob, run.classes, run.size = True, list(ck["classes"]), int(ck["imgsz"])
    run.mean = ck["mean"].flatten().numpy().astype(np.float32) * 255  # infer.py normalises 0-255 RGB pixels
    run.std = ck["std"].flatten().numpy().astype(np.float32) * 255
    return run


def load_classifier(path, cfg, device=None):
    if str(path).lower().endswith(".pt"):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ck, dict) and ck.get("arch") in ("swin_v2_t", "swin_v2_s", "swin_v2_b"):
            return load_species(ck, device)
    if str(path).lower().endswith((".pth", ".pt")):
        return load_pth(path, device)
    return load_onnx(path, cfg["onnx_providers"], int(cfg["onnx_preload_dlls"]))
