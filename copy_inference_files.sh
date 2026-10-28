#!/usr/bin/env bash
# Copy only what is needed for inference into ./mosquitto_msis (override with DEST=...)
set -euo pipefail
ROOT="/run/media/quy-linux/X10 Pro"
DEST="${DEST:-$ROOT/mosquitto_msis}"
PHAM="$ROOT/팜클/ultralytics_msys/ultralytics_msys/projects"
MM="$ROOT/엠시스랩전달본/mmlab/mmlab/mmpretrain/projects/phamcle_phase1_base2_invert"
APP="$ROOT/엠시스랩전달본/MosquitoIdentification_src/MosquitoIdentificationApp"

mkdir -p "$DEST"/{models,samples,reference}

# Stage 1: mosquito detector (YOLOv8s, imgsz 1520) and stage 2: body detector (YOLOv8n, imgsz 512)
cp -v "$PHAM/Mosquito/yolov8s_1520/weights/best.pt"      "$DEST/models/mosquito_det.pt"
cp -v "$PHAM/MosquitoBody/yolov8n_512/weights/best.pt"   "$DEST/models/mosquito_body_det.pt"
# Stage 3: classifier, SwinV2-B 384px, latest ("invert") set, epoch 66, ONNX
cp -v "$MM/deploy_epoch66/end2end.onnx"                  "$DEST/models/mosquito_cls.onnx"
cp -v "$APP/cls_labels.txt"                              "$DEST/models/cls_labels.txt"

# Reference only (training config, args) - not needed at runtime
cp -v "$MM/swinv2-base-w24_in21k-pre_384px.py"           "$DEST/reference/"
cp -v "$PHAM/Mosquito/yolov8s_1520/args.yaml"            "$DEST/reference/mosquito_det_args.yaml"
cp -v "$PHAM/MosquitoBody/yolov8n_512/args.yaml"         "$DEST/reference/mosquito_body_det_args.yaml"

# A few sample images (first 10 .jpg from the test data)
find "$ROOT/팜클/원본 데이터 및 labelme/테스트 데이터" -iname '*.jpg' | sort | head -10 | while read -r f; do cp -v "$f" "$DEST/samples/"; done

# Our code
cp -v "$(dirname "$0")/infer.py" "$(dirname "$0")/pyproject.toml" "$DEST/"
echo "Done -> $DEST"; du -sh "$DEST"
