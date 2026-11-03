---
marp: true
title: PhamCle Mosquito Detection & Classification Pipeline
paginate: true
---

# PhamCle Mosquito Detection & Classification System

### End-to-end pipeline analysis and architecture

- Re-implementation of the delivered MSIS Lab app pipeline (`infer.py`)
- Three models: detector → body detector → species classifier
- All runtime parameters in `config/config.yaml`

---

## Slide 1: Agenda

1. System overview and architecture
2. Stage 1: Input and preprocessing
3. Stage 2: Primary mosquito detection
4. Stage 3: Body detection and crop isolation
5. Stage 4: Classification and post-processing
6. Configuration, telemetry and debug outputs
7. Results on samples, limitations and next steps

---

## Slide 2: Problem & Goal

- **Input:** high-resolution trap photos (~9152 × 6944 px, ~63 MP)
- **Task:** find every mosquito and identify its species
- **Challenges:**
  - Mosquitoes are small relative to the frame
  - Specimens touch and overlap
  - Debris and non-mosquito insects are present
- **Output:** per-mosquito box, species label, confidence and annotated image

![[Insert Diagram/Image: Raw trap photo showing many small mosquitoes on the full 9152×6944 frame]]

---

## Slide 3: System Architecture

![[Insert Diagram/Image: Block diagram: Image → YOLOv8s detector → box merge → YOLOv8n body detector per crop → body merge → Voronoi split → SwinV2 classifier → threshold/top-2 → JSON + annotated image]]

| Stage | Component | Model / Method | Input size |
|---|---|---|---|
| 1 | Input & preprocessing | OpenCV load, letterbox resize | Full frame |
| 2 | Primary detection | YOLOv8s (`mosquito`), 11.1 M params | 1520 → 1536 px |
| 3 | Body detection | YOLOv8n (`mosquito_body`), 3.0 M params | 512 px per crop |
| 4 | Classification | SwinV2-Base, ONNX (4 classes) | 384 × 384 |

---

## Slide 4: Technology Stack

| Layer | Technology |
|---|---|
| Detectors | Ultralytics YOLOv8 (PyTorch, CUDA 12.6) |
| Classifier | mmpretrain SwinV2 → ONNX Runtime GPU |
| Image I/O | OpenCV, NumPy |
| Config | YAML (`config/config.yaml`), 38 parameters |
| Logging | loguru, size rotation + age retention |
| Hardware tested | NVIDIA RTX 4090 |

- The original app is C# WinForms with TensorRT engines
- This Python pipeline mirrors its logic; vendor SDK merge steps are re-implemented

---

## Slide 5: Stage 1: Input & Image Acquisition

- **Source:** single image file or a directory, scanned recursively
- **Formats:** `.jpg`, `.jpeg`, `.png`, configurable via `image_extensions`
- **Capture hardware:** Sentech industrial camera (SDK in the handover)
- Unreadable files are logged as warnings and skipped

![[Insert Diagram/Image: Example raw input image from samples/Ae.albopictus at full resolution]]

---

## Slide 6: Stage 1: Resolution & Normalization

| Model | Resize | Normalization |
|---|---|---|
| Detector | Letterbox to 1536 px (aspect kept) | Ultralytics default (0–1) |
| Body detector | Letterbox to 512 px per crop | Ultralytics default (0–1) |
| Classifier | Direct resize to 384×384 (aspect **not** kept) | RGB mean/std |

- Classifier mean (RGB): `123.675, 116.28, 103.53`
- Classifier std (RGB): `58.395, 57.12, 57.375`
- Values match the SwinV2 training config (`keep_ratio=False`, scale 384)

---

## Slide 7: Stage 1: ROI Extraction Strategy

- **No tiling:** the full frame goes to the detector in one pass
- **ROIs:** produced by the detector, then cropped from the **original** full-resolution image
- **Why:** crops keep the full sensor detail for body detection and classification
- Crops are clipped to the image bounds; empty crops are dropped

![[Insert Diagram/Image: Full frame with primary ROI rectangles, plus a zoomed inset of one full-resolution crop]]

---

## Slide 8: Stage 2: Primary Mosquito Detection

- **Model:** YOLOv8s, single class `mosquito`
- **Training:** 1520 px, 100 epochs, batch 64
- **Training augmentation:** mosaic 1.0, mixup 0.5, cutmix 0.1, rotation ±180°
- **Inference parameters:**
  - `det_conf` = 0.25 confidence
  - `det_iou` = 0.5 NMS IoU
- Ultralytics rounds 1520 up to 1536 (must be a multiple of the stride 32)

![[Insert Diagram/Image: Primary detection result: full frame with green YOLOv8s bounding boxes]]

---

## Slide 9: Stage 2: Overlap Box Merge

- **Problem:** NMS still leaves split or nested boxes around one mosquito
- **Merge rule:** union two boxes when *intersection ÷ smaller box area* > `merge_thr` (0.4)
- Repeats up to `merge_iter` passes (1); stops early when nothing changes
- The merged box keeps the highest score

| Step | Criterion | Threshold |
|---|---|---|
| NMS (YOLO) | IoU | 0.5 |
| Merge | Min-area overlap ratio | 0.4 |

![[Insert Diagram/Image: Before/after: two overlapping detector boxes merged into one union box]]

---

## Slide 10: Stage 3: Body Detection

- **Model:** YOLOv8n, single class `mosquito_body`, run inside each primary crop
- **Training:** 512 px, 100 epochs, same augmentation recipe as the detector
- **Scope:** finds the mosquito **body as one region**
  - No separate part classes (head, thorax, abdomen, wings, legs)
- **Purpose:**
  - Separate touching mosquitoes inside one crop
  - Exclude legs and wings that spill outside the body
- **Fallback:** if no body is found, the whole crop is classified

![[Insert Diagram/Image: One primary crop with orange body bounding boxes from YOLOv8n]]

---

## Slide 11: Stage 3: Body Merge (Two Levels)

- **Per crop:** same min-area merge as Stage 2 (`merge_thr` 0.4)
- **On the original image:** primary crops can overlap, so one body may be found twice
  - Bodies are sorted largest-first
  - A body is dropped if it overlaps a kept body by > `body_org_thr` (0.6)
- **Result:** one entry per physical mosquito body

![[Insert Diagram/Image: Full frame with final deduplicated body boxes after the original-scale merge]]

---

## Slide 12: Stage 3: Voronoi Crop Isolation

- **Case:** a primary crop contains several bodies
- **Method:** pixels closer to *another* body's centre are painted with the mean colour
- After normalisation those pixels become **0**, so they are neutral to the classifier
- Each body gets its own clean classifier input

![[Insert Diagram/Image: Cropped body-part ROIs: a multi-body crop and its per-body Voronoi-masked classifier inputs side by side]]

---

## Slide 13: Stage 4: Species Classification

- **Model:** SwinV2-Base (window 24/24/24/12), ImageNet-21k → KDCA → PhamCle fine-tune
- **Checkpoint:** `phamcle_phase1_base2_invert`, best top-1 at epoch 66
- **Loss:** Focal loss (α 0.4, γ 2.5)
- **Runtime:** ONNX, batched (`cls_batch` = 8), CUDA with CPU fallback

| Index | Label (KR) | Meaning |
|---|---|---|
| 0 | 얼룩날개모기 | *Anopheles* spp. |
| 1 | 작은빨간집모기 | *Culex tritaeniorhynchus* |
| 2 | 비모기 | Non-mosquito |
| 3 | 그 외 모기 | Other mosquito |

---

## Slide 14: Stage 4: Thresholding & Decision

- Softmax is applied when the ONNX output is logits (auto-detected)
- **Top-1** < `cls_thr` (0.1) → label `unknown`, score 0
- **Top-2** label and score are reported for review
- Low `cls_thr`: almost every crop gets a species label; raise it for stricter output

| Filter | Parameter | Default |
|---|---|---|
| Detection confidence | `det_conf` | 0.25 |
| Detector NMS | `det_iou` | 0.5 |
| Box merge | `merge_thr` | 0.4 |
| Body dedup | `body_org_thr` | 0.6 |
| Classification | `cls_thr` | 0.1 |

---

## Slide 15: Stage 4: Output & Telemetry

- **Annotated image** for each input in `output/`:
  - Green = primary box
  - Orange = body box
  - Label and score
- **`results.json` entry per detection:**
  - `det1` = primary box
  - `det2` = body box
  - `label`, `score`, `label2`, `score2`
- **Performance:** total ms per image
- **Logs:** `logging/infer_YYYYMMDD.log`, verbosity from `debug_enabled` and `debug_level`

![[Insert Diagram/Image: Final classified output overlay: frame with green/orange boxes and species labels with scores]]

---

## Slide 16: Debug Step Images

- Enabled by `debug_step_images` = 1
- Written to `output/steps/<frame>/`, named in pipeline order

| File | Content |
|---|---|
| `step_01_primary_boxes` | Frame + merged detector boxes |
| `step_02_roi_NN` | Each primary crop |
| `step_03_roi_NN_body_*` | Body boxes and body crops |
| `step_04_merged_bodies` | Bodies after the original-scale merge |
| `step_05_classifier_input_NN` | Voronoi-masked classifier inputs |

![[Insert Diagram/Image: Contact sheet of step_01 to step_05 images for one frame]]

---

## Slide 17: Configuration Management

- **One file:** `config/config.yaml` (38 parameters)
- **Each parameter has:**
  - Name and definition
  - Min, Max, Default, Best and Step
  - Current, the value the code reads
- **Parameter groups:**
  - Thresholds
  - Model sizes
  - Paths
  - Devices and ONNX providers
  - Normalization
  - Drawing
  - Logging and debug flags
- CLI `--models`, `--output` and `--device` override config paths
- **Benefit:** tune behaviour without editing code

---

## Slide 18: Sample Run Results

- **Set:** 92 images from `samples/Ae.albopictus` on an RTX 4090
- **Detections:** 220 mosquitoes in total

| Metric | Value |
|---|---|
| Mean latency | ~1.07 s / image |
| Min / max | 0.77 s / 4.71 s |
| First image (GPU warm-up) | 3.04 s |
| Labelled 작은빨간집모기 | 148 |
| Labelled 그 외 모기 | 72 |

- ⚠️ All samples are *Aedes albopictus*, so the expected label is **그 외 모기**
- Two-thirds were predicted as *Culex tritaeniorhynchus*, which needs investigation

---

## Slide 19: Limitations

- **Classifier scope:** only 2 named species; other species fall into "other"
- **Misclassification risk:** *Ae. albopictus* is often predicted as *Cx. tritaeniorhynchus*
- **Re-implemented merges:**
  - The merge and Voronoi steps come from parameter names, not vendor source
  - They match closely but are not bit-exact
- **No body-part model:** the body detector gives one region, not anatomical parts
- **Aspect distortion:** the classifier resize stretches crops (same as in training)
- **Shared threshold:** the body detector reuses `det_conf`

---

## Slide 20: Next Steps

- Validate on a labelled test set: per-class precision, recall and confusion matrix
- Retrain or extend the classifier with *Aedes* classes (the 5-class `phamcle` set exists)
- Tune `cls_thr` using the validation set
- Benchmark against the original TensorRT app output
- Consider TensorRT export for the Python pipeline to reduce latency
- Add a separate `body_conf` parameter if needed

---

## Slide 21: Summary

- **Three-stage cascade:**
  - Detect mosquito
  - Isolate the body
  - Classify the species
- Full-resolution crops keep fine morphology for the classifier
- Two-level merging and Voronoi masking handle touching specimens
- Fully config-driven, with traceable debug images and JSON telemetry
- **Key open issue:** classifier accuracy on *Aedes* specimens

![[Insert Diagram/Image: Compact one-line pipeline graphic with the final annotated output as the hero image]]
