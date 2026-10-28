"""Mosquito inference: detect -> body detect (inside each crop) -> classify (ONNX, SwinV2 384px)."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB, from the training config
STD = np.array([58.395, 57.12, 57.375], np.float32)
SIZE = 384


def classify(sess, labels, crop_bgr):
    x = cv2.resize(crop_bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    x = (cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
    x = x.transpose(2, 0, 1)[None]
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if not np.isclose(out.sum(), 1.0, atol=1e-3):  # logits -> softmax
        e = np.exp(out - out.max())
        out = e / e.sum()
    i = int(out.argmax())
    return labels[i], float(out[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--models", default="models")
    ap.add_argument("--output", default="output")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="e.g. cpu, 0")
    a = ap.parse_args()

    m = Path(a.models)
    det = YOLO(m / "mosquito_det.pt")
    body = YOLO(m / "mosquito_body_det.pt")
    labels = (m / "cls_labels.txt").read_text(encoding="utf-8").splitlines()
    labels = [s.strip() for s in labels if s.strip()]
    sess = ort.InferenceSession(str(m / "mosquito_cls.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        dets = []
        for b in det.predict(img, imgsz=1520, conf=a.conf, iou=0.5, device=a.device, verbose=False)[0].boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = b
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bb = body.predict(crop, imgsz=512, conf=a.conf, device=a.device, verbose=False)[0].boxes
            if len(bb):  # tightest body crop, else fall back to the detection crop
                bx1, by1, bx2, by2 = bb.xyxy[int(bb.conf.argmax())].cpu().numpy().astype(int)
                crop = crop[by1:by2, bx1:bx2]
            label, score = classify(sess, labels, crop)
            dets.append({"box": [int(x1), int(y1), int(x2), int(y2)], "label": label, "score": round(score, 4)})
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"{label} {score:.2f}", (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(out / f.name), img)
        results.append({"file": str(f), "detections": dets})
        print(f.name, [(d["label"], d["score"]) for d in dets])
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
