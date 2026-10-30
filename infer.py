"""Mosquito inference, following the delivered app's pipeline (defaults = App.config values):

  detector (YOLOv8s, 1520) -> merge overlapping boxes -> body detector (YOLOv8n, 512) inside each crop
  -> merge overlapping bodies (per crop, then on the original image) -> Voronoi split when a crop holds
  several bodies -> classifier (SwinV2 384px, ONNX) -> top-2 with a classification threshold.

The merge / Voronoi steps live in the vendor's native SDK, so they are re-implemented here from the
parameter names in PhamcleConfig.cs and Docs/handover; they are a close match, not bit-exact.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

MEAN = np.array([123.675, 116.28, 103.53], np.float32)  # RGB, from the training config
STD = np.array([58.395, 57.12, 57.375], np.float32)
SIZE = 384
FILL_RGB = MEAN  # masked-out (other mosquitos') pixels become 0 after normalisation


def overlap_ratio(a, b):
    """Intersection area / area of the smaller box (the app's 'MinArea overlap ratio')."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])), 1e-9)


def merge_boxes(boxes, scores, thr, iterations):
    """Union boxes whose overlap ratio exceeds thr; repeat `iterations` times. Keeps the max score."""
    boxes = [list(map(float, b)) for b in boxes]
    scores = list(scores)
    for _ in range(iterations):
        used = [False] * len(boxes)
        nb, ns = [], []
        for i in range(len(boxes)):
            if used[i]:
                continue
            cur, sc = boxes[i], scores[i]
            for j in range(i + 1, len(boxes)):
                if not used[j] and overlap_ratio(cur, boxes[j]) > thr:
                    cur = [min(cur[0], boxes[j][0]), min(cur[1], boxes[j][1]),
                           max(cur[2], boxes[j][2]), max(cur[3], boxes[j][3])]
                    sc = max(sc, scores[j])
                    used[j] = True
            used[i] = True
            nb.append(cur)
            ns.append(sc)
        if len(nb) == len(boxes):
            break
        boxes, scores = nb, ns
    return boxes, scores


def voronoi_crop(crop, body_boxes, k):
    """Crop body k out of `crop`; pixels nearer to another body's centre are painted out (Voronoi cell of k)."""
    x1, y1, x2, y2 = body_boxes[k]
    sub = crop[y1:y2, x1:x2].copy()
    if len(body_boxes) == 1 or sub.size == 0:
        return sub
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in body_boxes], np.float32)
    ys, xs = np.mgrid[y1:y2, x1:x2]
    d = (xs[..., None] - centres[:, 0]) ** 2 + (ys[..., None] - centres[:, 1]) ** 2
    sub[d.argmin(-1) != k] = FILL_RGB[::-1].astype(np.uint8)  # BGR
    return sub


def classify(sess, crops_bgr):
    """Batch of BGR crops -> (N, num_classes) probabilities."""
    x = np.stack([(cv2.cvtColor(cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                   .astype(np.float32) - MEAN) / STD for c in crops_bgr]).transpose(0, 3, 1, 2)
    out = sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x)})[0]
    if not np.allclose(out.sum(1), 1.0, atol=1e-3):  # logits -> softmax
        e = np.exp(out - out.max(1, keepdims=True))
        out = e / e.sum(1, keepdims=True)
    return out


def xywh(b):
    return {"x": int(b[0]), "y": int(b[1]), "w": int(b[2] - b[0]), "h": int(b[3] - b[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--models", default="models")
    ap.add_argument("--output", default="output")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0")
    ap.add_argument("--det-conf", type=float, default=0.25, help="DetectionThreshold")
    ap.add_argument("--cls-thr", type=float, default=0.1, help="ClassificationThreshold: below it the label is 'unknown'")
    ap.add_argument("--merge-thr", type=float, default=0.4, help="MergeOverlapThresholdMosquito[Body]")
    ap.add_argument("--merge-iter", type=int, default=1, help="MergeIterationsMosquito[Body]")
    ap.add_argument("--body-org-thr", type=float, default=0.6, help="MosquitoBodyMergeOnOrgScaleThreshold")
    ap.add_argument("--cls-batch", type=int, default=8)
    a = ap.parse_args()

    m = Path(a.models)
    det = YOLO(m / "mosquito_det.pt")
    body = YOLO(m / "mosquito_body_det.pt")
    labels = [s.strip() for s in (m / "cls_labels.txt").read_text(encoding="utf-8").splitlines() if s.strip()]
    try:
        ort.preload_dlls()  # find pip-installed CUDA/cuDNN libs
    except Exception:
        pass
    sess = ort.InferenceSession(str(m / "mosquito_cls.onnx"),
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        t0 = time.time()
        H, W = img.shape[:2]

        # Stage 1: mosquito detector + merge
        r = det.predict(img, imgsz=1520, conf=a.det_conf, iou=0.5, device=a.device, verbose=False)[0].boxes
        boxes, _ = merge_boxes(r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), a.merge_thr, a.merge_iter)

        # Stage 2: body detector inside each crop, merged per crop
        items = []  # one per body: det1 box, body box (full-image coords), crop-relative geometry
        for b in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bb = body.predict(crop, imgsz=512, conf=a.det_conf, device=a.device, verbose=False)[0].boxes
            bxs, bsc = merge_boxes(bb.xyxy.cpu().numpy(), bb.conf.cpu().numpy(), a.merge_thr, a.merge_iter)
            if not bxs:  # no body found: classify the whole detection crop
                bxs, bsc = [[0, 0, x2 - x1, y2 - y1]], [0.0]
            bxs = [[int(max(v, 0)) for v in bx] for bx in bxs]
            for k, bx in enumerate(bxs):
                items.append({"det1": [x1, y1, x2, y2], "body": [bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1],
                              "crop": crop, "rel": bxs, "k": k})

        # Merge bodies again on the original image (the same mosquito can appear in overlapping det crops)
        keep = []
        for it in sorted(items, key=lambda i: -(i["body"][2] - i["body"][0]) * (i["body"][3] - i["body"][1])):
            if all(overlap_ratio(it["body"], o["body"]) <= a.body_org_thr for o in keep):
                keep.append(it)
        items = keep

        # Stage 3: Voronoi-split crops -> classifier (batched)
        crops = [voronoi_crop(i["crop"], i["rel"], i["k"]) for i in items]
        probs = np.concatenate([classify(sess, crops[s:s + a.cls_batch]) for s in range(0, len(crops), a.cls_batch)]) \
            if crops else np.zeros((0, len(labels)))

        dets = []
        for it, p in zip(items, probs):
            order = np.argsort(-p)
            i1 = int(order[0])
            i2 = int(order[1]) if len(order) > 1 else None
            known = p[i1] >= a.cls_thr
            d = {"det1": xywh(it["det1"]), "det2": xywh(it["body"]),
                 "label": labels[i1] if known else "unknown", "score": round(float(p[i1]), 4) if known else 0.0}
            if i2 is not None and known:
                d.update(label2=labels[i2], score2=round(float(p[i2]), 4))
            dets.append(d)
            sc = max(W, H) / 1600  # scale drawing to the image size so it stays visible on 9k-px frames
            lw = max(2, round(2 * sc))
            cv2.rectangle(img, (it["det1"][0], it["det1"][1]), (it["det1"][2], it["det1"][3]), (0, 255, 0), lw)  # stage 1
            cv2.rectangle(img, (it["body"][0], it["body"][1]), (it["body"][2], it["body"][3]), (255, 128, 0), lw)  # stage 2
            txt = f"{d['label']} {float(p[i1]):.2f}"  # stage 3
            fs, th = 0.6 * sc, max(1, round(sc))
            (tw, tgh), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx, ty = it["det1"][0], max(it["det1"][1] - 6, tgh + 4)
            cv2.rectangle(img, (tx, ty - tgh - 4), (tx + tw, ty + bl), (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 0), th)
        cv2.imwrite(str(out / f.name), img)
        ms = int((time.time() - t0) * 1000)
        results.append({"file": str(f), "detections": dets, "performance": {"total": ms}})
        print(f.name, f"{ms}ms", [(d["label"], d["score"]) for d in dets])
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
