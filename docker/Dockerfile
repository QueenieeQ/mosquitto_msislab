# Mosquito inference (infer.py) - GPU image. Build context = project root:
#   docker build -f docker/Dockerfile -t mosquito-msis .
# Model weights (*.pt / *.onnx) are NOT baked in; mount ./models at run time.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    PYTHONUNBUFFERED=1

# opencv-python needs libGL / glib
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependencies first (cached layer); torch comes from the cu126 index pinned in pyproject.toml
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 && uv sync --frozen --no-install-project --python 3.12

COPY infer.py ./
COPY config ./config
# cls_labels.txt is the only tracked file in models/
COPY models/cls_labels.txt ./models/cls_labels.txt

ENTRYPOINT ["python", "infer.py"]
CMD ["--input", "samples"]
