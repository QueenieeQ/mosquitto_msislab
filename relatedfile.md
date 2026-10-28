# Mosquito Classification Project — Related Files on X10 Pro

Generated 2026-10-28. Root: `/run/media/quy-linux/X10 Pro/`
(File counts are recursive and include images + label files.)

## 1. Summary

| # | Top-level folder | Relevance | Notes |
|---|---|---|---|
| 1 | `엠시스랩전달본/` (MSYS Lab handover) | **Core** | Source app, training code, datasets, docs |
| 2 | `팜클/` (Phamcle) | **Core** | Raw/labelme data, ultralytics copy, training history, final report |
| 3 | `자가학습/` (self-learning) | Probable | `yolo_multi_train.py`, `config/` — verify |
| 4 | `수인산업/` | Unclear | `1차분`–`7차분`, empty `models/` and `배포 소프트웨어/` — verify |
| 5 | `Mosquitto_MSIS/` | This file only | Note: spelled "Mosquitto" (MQTT broker name), not "Mosquito" |
| 6 | `digitaltwin/`, `심토스/` | Probably unrelated | Unity digital-twin build / MSISLAB installer |
| 7 | `SentechSDK/`, `DL_PC/` | Supporting | Camera SDK; Anaconda/VS Code/Cursor installers |

## 2. `엠시스랩전달본/` — handover package

Per `데이터셋 위치 관련 설명.txt`: the `mmlab/mmpretrain/data` directory containing `invert` is the most recently used training set.

### Top-level items
- `MosquitoIdentification_src/` (+ `.zip`) — C# WinForms app, 17,488 files
- `docs/` (+ `docs.zip`) — design doc v2, workflow v1, maintenance history v1.1 (`.docx`)
- `mmlab/` (+ `mmlab.tar.gz`) — mmpretrain / mmdetection / mmdeploy
- `ultralytics/` (+ `ultralytics.tar.gz`) — YOLO training code and datasets
- `모기 종별 형태학적 특징 ver 3_ 20251027.pptx` — species morphology slides
- `데이터셋 위치 관련 설명.txt`

### Application: `MosquitoIdentification_src/`
- `MosquitoIdentification.sln`, `MosquitoIdentificationApp/MosquitoIdentificationApp.csproj`
- `Models/IMosquitoSpeciesProvider.cs`, `Models/MosquitoData.cs`
- `Services/MosquitoSpeciesService.cs`
- `x64/Debug/` and `x64/Release/` — built exe plus 13 model files each (TensorRT `.engine` / `.onnx`):
  - Classifiers: `phamcle_phase1`, `_phase1_base` (epoch 39, 91), `_phase1_base2` (epoch 58, 96), `_phase1_base2_invert` (epoch 61, 66), `_phase1_epoch59`
  - Detectors: `phamcle_phase1_mosquito_det_nms`, `phamcle_phase1_mosquito_body_det_nms`

### Classification datasets: `mmlab/mmlab/mmpretrain/data/`
| Folder | Files | Species classes |
|---|---:|---|
| `phamcle` | 6,538 | 5: Aedes albopictus, Aedes vexans, Anopheles sinensis, Culex pipiens, Culex tritaeniorhynchus |
| `phamcle_phase1` | 2,161 | 2: Anopheles, Culex tritaeniorhynchus |
| `phamcle_phase1_base` | 2,150 | 2 (same) |
| `phamcle_phase1_base2` | 2,151 | 2 (same) |
| **`phamcle_phase1_base2_invert`** | 2,150 | 2 (same) — **latest per handover note** |
| `phamcle_test`, `_test2`, `_test3`, `_test4` | 1,213 / 1,300 / 1,213 / 1,360 | Culex tritaeniorhynchus |
| `phamcle_camera_setup` | 27 | aedes (test) |
| `KDCA` | 1,462 | aedes (train / train_kfold / validation) |

Each has `train/`, `validation/`, `test/`, `test2/` splits where applicable.

### Trained model projects: `mmlab/mmlab/mmpretrain/projects/`
- `phamcle_phase1_base/deploy_epoch39/end2end.onnx`
- `phamcle_phase1_base2_invert_smoothloss/deploy_epoch61/end2end.onnx`
- `phamcle_phase1_swinv2/deploy~/end2end.onnx`

### Detection (mmdetection)
- `mmlab/mmlab/mmdetection/datasets/Mosquito` — 1,322 files
- `mmlab/mmlab/mmdetection/projects/Mosquito` — 5 files
- `mmlab/mmlab/mmdeploy/rtmpose-ort/rtmpose-m/end2end.onnx`

### YOLO / ultralytics: `ultralytics/ultralytics/`
Scripts: `mosquito_train.py`, `mosquito_kfold_train.py`, `mosquito_prediction.py`, `filter_multi_mosquito_body.py`, `phamcle_pipeline.py`, `yolo2labelme.py`, `convert2yolo.py`, `copy_labels_to_folds.py`, `find_common_labels.py`

| Dataset (`datasets/`) | Files |
|---|---:|
| `Mosquito` | 22,700 |
| `MosquitoBody` | 15,208 |
| `Mosquito_kfold` | 6,605 |
| `Mosquito_body_kfold` | 6,605 |
| `MosquitoBodyPose` | 716 |

Inference output: `inference/Mosquito`, `inference/MosquitoBody` (`MosquitoBody_val`, `MosquitoBody_val_mixup0.5_cutmix0.1`).
Not mosquito-related, same tree: Barcode, Bump, FASDD, Forklift, safety-gear, KDCA (YOLO side), `ats_`/`hands_`/`idis_`/`fasdd_`/`bump_`/`barcode_`/`safety_gear_` scripts.

## 3. `팜클/` — Phamcle project

- `classification 학습 히스토리.docx` — classification training history (`~$…docx` is a stale Word lock file, safe to delete)
- `과제완료보고서.pptx` — project completion report
- `ultralytics_msys/` (+ `.tar.gz`) — working copy of the YOLO code
  - `projects/Mosquito/yolov8s_1520`, `projects/MosquitoBody/yolov8n_512` (and `yolov8n_512~`)
  - `inference/Mosquito`, `inference/MosquitoBody`
  - extra scripts vs. handover copy: `adjust_labelme_bboxes.py`, `find_false_prediction.py`, `barcode_auto_anno.py`
- `원본 데이터 및 labelme/` — raw images + labelme annotations

| Folder | Files |
|---|---:|
| `1차 모기 detection` | 527 |
| `2차 모기몸통 detection` | 2,485 |
| `3차 모기 classification` | 2,458 |
| `테스트 데이터` | 470 |
| `yolo_train_config.json` | 1 |

`3차 모기 classification` is split into `주요 2종` (Anopheles; Culex tritaeniorhynchus) and `그 외` (Aedes albopictus, Aedes vexans niponii, Aedes spp., Culex pipiens, Culex spp.), each with a `검증` (validation) copy.

## 4. Other folders to verify
- `자가학습/` — `yolo_multi_train.py`, `config/` (name suggests self-learning training)
- `수인산업/` — `1차분`…`7차분` batches; `models/` and `배포 소프트웨어/` are empty
- `SentechSDK/v1_2` — Sentech camera SDK (likely the capture hardware)
- `digitaltwin/`, `심토스/` — Unity digital-twin (SIMTOS 2026, MSISLAB); likely unrelated

## 5. Observations for review
1. **Duplicates**: `ultralytics/`, `mmlab/`, `docs/`, `MosquitoIdentification_src/` each exist as both a folder and an archive (`.zip` / `.tar.gz`), roughly doubling storage.
2. **Overlapping copies of ultralytics code**: `엠시스랩전달본/ultralytics` and `팜클/ultralytics_msys` — the latter has newer scripts.
3. **Phase-1 dataset variants** (`phase1`, `base`, `base2`, `base2_invert`) differ by only ~11 files; the real differences are likely in augmentation/inversion, not size.
4. **Build artifacts** (`x64/Debug` + `x64/Release`) hold duplicated model files.
5. **Sizes** (from `du`):

| Item | Folder | Archive |
|---|---:|---:|
| `엠시스랩전달본/mmlab` | 189 GB | 159 GB |
| `엠시스랩전달본/ultralytics` | 161 GB | 87 GB |
| `엠시스랩전달본/MosquitoIdentification_src` | 25 GB | 12 GB |
| `엠시스랩전달본/docs` | 3.3 MB | 2.3 MB |
| `팜클/원본 데이터 및 labelme` | 19 GB | — |
| `팜클/ultralytics_msys` | 16 GB | 8.9 GB |
| `수인산업/1차분`–`4차분` | 7.0 / 6.9 / 3.8 / 3.2 GB | — |
| `수인산업/5차분`–`7차분` | empty (256 KB) | — |
| `심토스` | 3.7 GB | — |

   The mosquito-related archives total about 267 GB, and each one duplicates a folder that is still on the disk.
