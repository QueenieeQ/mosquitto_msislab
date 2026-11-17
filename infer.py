"""Mosquito inference, following the delivered app's pipeline (defaults = App.config values):

  detector (YOLOv8s, 1520) -> merge overlapping boxes -> body detector (YOLOv8n, 512) inside each crop
  -> merge overlapping bodies (per crop, then on the original image) -> Voronoi split when a crop holds
  several bodies -> classifier (SwinV2 384px, ONNX) -> top-2 with a classification threshold.

The merge / Voronoi steps live in the vendor's native SDK, so they are re-implemented here from the
parameter names in PhamcleConfig.cs and Docs/handover; they are a close match, not bit-exact.
"""
import argparse
import json
import os
import time
from pathlib import Path

import yaml
from loguru import logger

import cv2
import numpy as np
from ultralytics import YOLO

from cls_backend import load_classifier

BASE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("INFER_CONFIG", BASE / "config" / "config.yaml"))  # override for batch evaluation
# debug_level (0-7) -> minimum loguru level; higher = more verbose
LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "TRACE", "TRACE", "TRACE"]


def load_config(path=CONFIG_PATH):
    """Read config.yaml; returns {param: Current}."""
    with open(path, encoding="utf-8") as f:
        return {k: v["Current"] for k, v in yaml.safe_load(f).items()}


def resolve(p):
    """Relative paths in config.yaml / CLI are taken relative to this file's directory."""
    p = Path(p)
    return p if p.is_absolute() else BASE / p


def setup_logging(cfg):
    """File sink in log_dir (created if missing), size rotation + age retention.

    debug_enabled=0 caps the sink at INFO; debug_enabled=1 lets debug_level choose the threshold."""
    log_dir = resolve(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    enabled = int(cfg["debug_enabled"])
    level = LEVELS[min(max(int(cfg["debug_level"]), 0), len(LEVELS) - 1)]
    threshold = logger.level(level).no if enabled else logger.level("INFO").no
    logger.add(log_dir / "infer_{time:YYYYMMDD}.log", level="TRACE",
               filter=lambda r: r["level"].no >= threshold,
               rotation=f"{int(cfg['log_rotation_mb'])} MB", retention=f"{int(cfg['log_retention_days'])} days",
               encoding="utf-8")


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


def voronoi_crop(crop, body_boxes, k, fill_rgb):
    """Crop body k out of `crop`; pixels nearer to another body's centre are painted out (Voronoi cell of k)."""
    x1, y1, x2, y2 = body_boxes[k]
    sub = crop[y1:y2, x1:x2].copy()
    if len(body_boxes) == 1 or sub.size == 0:
        return sub
    centres = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in body_boxes], np.float32)
    ys, xs = np.mgrid[y1:y2, x1:x2]
    d = (xs[..., None] - centres[:, 0]) ** 2 + (ys[..., None] - centres[:, 1]) ** 2
    sub[d.argmin(-1) != k] = fill_rgb[::-1].astype(np.uint8)  # BGR
    return sub


def classify(run, crops_bgr, size, mean, std, softmax_atol):  # run.is_prob: already probabilities (sigmoid), keep as is
    """Batch of BGR crops -> (N, num_classes) probabilities."""
    x = np.stack([(cv2.cvtColor(cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                   .astype(np.float32) - mean) / std for c in crops_bgr]).transpose(0, 3, 1, 2)
    out = run(x)
    if not getattr(run, "is_prob", False) and not np.allclose(out.sum(1), 1.0, atol=softmax_atol):  # logits -> softmax
        e = np.exp(out - out.max(1, keepdims=True))
        out = e / e.sum(1, keepdims=True)
    return out


def draw_boxes(img, boxes, color, lw):
    """Copy of img with boxes (x1, y1, x2, y2) outlined."""
    img = img.copy()
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, lw)
    return img


def xywh(b):
    return {"x": int(b[0]), "y": int(b[1]), "w": int(b[2] - b[0]), "h": int(b[3] - b[1])}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Tunable parameters come from config.yaml (Current values)")
    ap.add_argument("--input", required=True, help="image file or directory")
    ap.add_argument("--models", default=cfg["models_dir"])
    ap.add_argument("--output", default=cfg["output_dir"])
    ap.add_argument("--cls-model", default=cfg["cls_model"],
                    help="classifier file: .onnx or an mmpretrain .pth checkpoint; relative to --models unless absolute")
    ap.add_argument("--device", default=cfg["device"], help="e.g. cpu, 0")
    a = ap.parse_args()
    setup_logging(cfg)
    logger.info("config: {}", cfg)

    m = resolve(a.models)
    mean = np.array(cfg["cls_mean"], np.float32)
    std = np.array(cfg["cls_std"], np.float32)
    fill_rgb = mean  # masked-out (other mosquitos') pixels become 0 after normalisation
    exts = {e.lower() for e in cfg["image_extensions"]}
    det = YOLO(m / cfg["det_model"])
    body = YOLO(m / cfg["body_model"])
    labels = [s.strip() for s in (m / cfg["cls_labels"]).read_text(encoding="utf-8").splitlines() if s.strip()]
    cls_path = Path(a.cls_model)
    cls_path = cls_path if cls_path.is_absolute() else m / cls_path
    run_cls = load_classifier(cls_path, cfg, a.device)
    logger.info("classifier: {}", cls_path)
    cls_size = int(cfg["cls_size"])
    if hasattr(run_cls, "classes"):  # species checkpoint: its own labels, normalisation and input size
        labels, mean, std, cls_size = run_cls.classes, run_cls.mean, run_cls.std, run_cls.size
        fill_rgb = mean

    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    out = resolve(a.output)
    out.mkdir(parents=True, exist_ok=True)
    save_steps = int(cfg["debug_step_images"]) == 1
    steps_root = out / "steps"
    if save_steps:
        steps_root.mkdir(parents=True, exist_ok=True)
    results = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            logger.warning("cannot read {}", f)
            continue
        t0 = time.time()
        logger.debug("processing {}", f)
        H, W = img.shape[:2]
        # Intermediate images go to <output>/steps/<frame>/step_NN_*.jpg (only when debug_step_images = 1)
        steps = steps_root / f.stem
        lw_dbg = max(2, round(cfg["draw_line_width"] * max(W, H) / cfg["draw_ref_size"]))

        def save_step(name, im):
            if save_steps:
                steps.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(steps / f"{name}.jpg"), im)

        # Stage 1: mosquito detector + merge
        r = det.predict(img, imgsz=int(cfg['det_imgsz']), conf=cfg['det_conf'], iou=cfg['det_iou'], device=a.device, verbose=False)[0].boxes
        boxes, _ = merge_boxes(r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), cfg['merge_thr'], int(cfg['merge_iter']))

        if save_steps:
            save_step("step_01_primary_boxes", draw_boxes(img, boxes, tuple(cfg["det1_color"]), lw_dbg))

        # Stage 2: body detector inside each crop, merged per crop
        items = []  # one per body: det1 box, body box (full-image coords), crop-relative geometry
        for n, b in enumerate(boxes, 1):
            x1, y1, x2, y2 = (int(round(v)) for v in b)
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            bb = body.predict(crop, imgsz=int(cfg['body_imgsz']), conf=cfg['det_conf'], device=a.device, verbose=False)[0].boxes
            bxs, bsc = merge_boxes(bb.xyxy.cpu().numpy(), bb.conf.cpu().numpy(), cfg['merge_thr'], int(cfg['merge_iter']))
            save_step(f"step_02_roi_{n:02d}", crop)
            logger.trace("crop {} -> {} bodies", [x1, y1, x2, y2], len(bxs))
            if not bxs:  # no body found: classify the whole detection crop
                bxs, bsc = [[0, 0, x2 - x1, y2 - y1]], [0.0]
            bxs = [[int(max(v, 0)) for v in bx] for bx in bxs]
            if save_steps:
                save_step(f"step_03_roi_{n:02d}_body_boxes", draw_boxes(crop, bxs, tuple(cfg["det2_color"]), lw_dbg))
                for k, bx in enumerate(bxs, 1):
                    save_step(f"step_03_roi_{n:02d}_body_{k:02d}", crop[bx[1]:bx[3], bx[0]:bx[2]])
            for k, bx in enumerate(bxs):
                items.append({"det1": [x1, y1, x2, y2], "body": [bx[0] + x1, bx[1] + y1, bx[2] + x1, bx[3] + y1],
                              "crop": crop, "rel": bxs, "k": k})

        # Merge bodies again on the original image (the same mosquito can appear in overlapping det crops)
        keep = []
        for it in sorted(items, key=lambda i: -(i["body"][2] - i["body"][0]) * (i["body"][3] - i["body"][1])):
            if all(overlap_ratio(it["body"], o["body"]) <= cfg['body_org_thr'] for o in keep):
                keep.append(it)
        items = keep
        if save_steps:
            save_step("step_04_merged_bodies", draw_boxes(img, [i["body"] for i in items], tuple(cfg["det2_color"]), lw_dbg))

        # Stage 3: Voronoi-split crops -> classifier (batched)
        crops = [voronoi_crop(i["crop"], i["rel"], i["k"], fill_rgb) for i in items]
        for n, c in enumerate(crops, 1):
            save_step(f"step_05_classifier_input_{n:02d}", c)
        probs = np.concatenate([classify(run_cls, crops[s:s + int(cfg['cls_batch'])], cls_size, mean, std, cfg['softmax_atol']) for s in range(0, len(crops), int(cfg['cls_batch']))]) \
            if crops else np.zeros((0, len(labels)))

        dets = []
        for it, p in zip(items, probs):
            order = np.argsort(-p)
            i1 = int(order[0])
            i2 = int(order[1]) if len(order) > 1 else None
            known = p[i1] >= cfg['cls_thr']
            d = {"det1": xywh(it["det1"]), "det2": xywh(it["body"]),
                 "label": labels[i1] if known else "unknown", "score": round(float(p[i1]), 4) if known else 0.0,
                 "probs": [round(float(x), 4) for x in p]}  # full softmax, in cls_labels.txt order
            if i2 is not None and known:
                d.update(label2=labels[i2], score2=round(float(p[i2]), 4))
            dets.append(d)
            sc = max(W, H) / cfg["draw_ref_size"]  # scale drawing to the image size so it stays visible on 9k-px frames
            lw = max(2, round(cfg["draw_line_width"] * sc))
            cv2.rectangle(img, (it["det1"][0], it["det1"][1]), (it["det1"][2], it["det1"][3]), tuple(cfg["det1_color"]), lw)  # stage 1
            cv2.rectangle(img, (it["body"][0], it["body"][1]), (it["body"][2], it["body"][3]), tuple(cfg["det2_color"]), lw)  # stage 2
            txt = f"{d['label']} {float(p[i1]):.2f}"  # stage 3
            fs, th = cfg["draw_font_scale"] * sc, max(1, round(sc))
            (tw, tgh), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx, ty = it["det1"][0], max(it["det1"][1] - int(cfg["draw_label_offset"]), tgh + 4)
            cv2.rectangle(img, (tx, ty - tgh - 4), (tx + tw, ty + bl), tuple(cfg["label_bg_color"]), -1)
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, tuple(cfg["label_text_color"]), th)
        cv2.imwrite(str(out / f.name), img)
        ms = int((time.time() - t0) * 1000)
        results.append({"file": str(f), "detections": dets, "performance": {"total": ms}})
        logger.info("{} {}ms {}", f.name, ms, [(d["label"], d["score"]) for d in dets])
        print(f.name, f"{ms}ms", [(d["label"], d["score"]) for d in dets])
    (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
