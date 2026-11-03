# GCP Migration Plan — Mosquito Detection & Classification Pipeline

Status: draft v1 · Written 2026-11-03 · Owner: project owner
Scope: make this project easy to move to Google Cloud Platform later. Phase 1 needs no GCP account and can start now.

## 1. Where we are today (facts from the repo)

| Area | Current state | Migration impact |
|---|---|---|
| Workload | Batch CLI: `infer.py` (256 lines), `find_bad_samples.py` (205 lines). Three stages: YOLOv8s detector (1520 px) → YOLOv8n body detector (512 px) → SwinV2-B classifier (ONNX, 384 px) | Batch fits Cloud Run Jobs / Vertex AI / GCE. No always-on server needed |
| Compute | One RTX 4090 (24 GB), CUDA 12 stack | Nearest cloud GPU is an NVIDIA L4 (24 GB). Verify availability per region |
| Python env | `uv` + `pyproject.toml` + `uv.lock`; torch from `pytorch-cu126` index; `onnxruntime-gpu>=1.22,<1.24` (CUDA 12 builds) | Good: reproducible. The image must use CUDA 12.x to match |
| Config | `config/config.yaml` (parameter → `Current` value), loaded by `infer.py` | Needs env/CLI override so one image runs in any environment |
| Paths | `infer.py` resolves paths relative to its own folder (portable). `copy_inference_files.sh` hardcodes an external drive (`/run/media/quy-linux/X10 Pro/...`) | Script is workstation-only; replace with a model manifest |
| Packaging | `docker/` is empty. No Dockerfile, no CI | Must create |
| Models | `models/` 363 MB (`.pt`, `.onnx`, labels); `classification/*.pth` 389 MB | Do not store in git or the image. Version them in GCS |
| Data | `data/`: Hyseop_data 228 GB, Pharmcle Mosquito 6.9 GB (+ 6.9 GB zip), 2nd_data 9.6 GB. Local outputs: `output/` 1.6 GB, `bad_samples/` 2.6 GB | Biggest cost and time item. Migrate only what the cloud needs (see §3) |
| Fidelity | Merge/Voronoi steps re-implement a vendor SDK, "a close match, not bit-exact" | Need golden outputs to prove cloud results match local |
| Training | Done elsewhere (mmpretrain / ultralytics, on the handover drive) | Out of scope here. Treat as a later phase (Vertex AI custom training) |

## 2. Strategy

1. **Make it cloud-ready locally first.** Containerize, externalize paths and config, and abstract storage. Every change must still run on this workstation, so risk stays low and the cloud choice stays open.
2. **Lift-and-shift first, optimize later.** The first cloud target is the closest match to today (one GPU VM or one GPU job running the same container). Move to managed services only after parity is proven.
3. **Separate the four things that change at different speeds:** code (git), container image (Artifact Registry), models (GCS, versioned), data and outputs (GCS). Nothing is baked into the image except code and dependencies.
4. **Prove parity before cutting over.** A frozen golden set of images with expected outputs, compared with a tolerance, is the exit gate for every phase.
5. **Control cost from day one.** Budgets and alerts before the first GPU hour. Spot/preemptible and scale-to-zero where the job is restartable.
6. **Avoid lock-in where it is cheap to.** Use plain containers, `gs://` through one small storage module, and env-var config. No GCP-only SDK calls scattered through the pipeline code.

## 3. Decisions to make first

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | What runs in the cloud? | Inference only / inference + training / everything | Inference first. Training later |
| D2 | How is it triggered? | Manual CLI / schedule / on file upload / HTTP API | Start manual (job run). Add an upload trigger later |
| D3 | First compute target | GCE GPU VM (closest to today) / Cloud Run Jobs with GPU / Vertex AI custom job | GCE L4 VM for parity. Then Cloud Run Jobs or Vertex for batch. Verify current GPU support and quota per region |
| D4 | Which data goes to the cloud? | All 245 GB / only what the cloud workload reads | Only what is needed. Archive the rest. The 228 GB Hyseop_data moves only if you train in the cloud |
| D5 | Region | Nearest to users vs GPU availability and price | Choose by GPU availability and data-residency rules, then keep all resources in it |
| D6 | Data governance | Is the Pharmcle/MSYS data allowed in a public cloud? | **Confirm with the data owner or contract before uploading anything** |
| D7 | Infra as code | Console clicks / `gcloud` scripts / Terraform | `gcloud` scripts in the repo now. Terraform once the shape is stable |

## 4. Phased plan

### Phase 0 — Assess (½ day, no GCP needed)
- [ ] Record the decisions D1–D7 above in this file.
- [ ] Inventory the data: what each folder under `data/` is, who owns it, whether it is confidential, whether the cloud needs it.
- [ ] Record the baseline on this machine: images/sec, GPU memory use, wall time for the `samples/` set, and the output files produced.
- [ ] Estimate cloud cost with the GCP pricing calculator (GPU hours, storage, egress). Do not rely on rough numbers from memory.

### Phase 1 — Make the code portable (2–4 days, no GCP needed)
1. **Golden set**
   - [ ] Pick 30–50 representative images from `samples/` (include `bad_samples/` cases).
   - [ ] Run `infer.py` and save the JSON outputs as `tests/golden/`. Define the tolerance (box IoU ≥ 0.95, same top-1 class, score ±0.01).
   - [ ] Add a small pytest that compares a fresh run to the golden outputs.
2. **Externalize paths and config**
   - [ ] Add env overrides: `MODEL_DIR`, `INPUT_URI`, `OUTPUT_URI`, `CONFIG_PATH`, `LOG_LEVEL`. CLI flags win over env, env wins over `config.yaml`.
   - [ ] Keep `config.yaml` defaults unchanged so local behavior does not change.
3. **Storage abstraction**
   - [ ] One module (`storage.py`) that reads and writes either a local path or `gs://`, for example via `fsspec` + `gcsfs`. Pipeline code only calls this module.
   - [ ] Write outputs to a temp dir, then upload at the end. This gives atomic results and safe retries.
4. **Logging and exit codes**
   - [ ] Log to stdout in structured form (JSON) so Cloud Logging parses it. Keep the file log optional.
   - [ ] Non-zero exit on failure, zero on success. Make reruns idempotent (skip images whose output already exists).
5. **Container**
   - [ ] `docker/Dockerfile`: CUDA 12.x runtime base, install `uv`, `uv sync --frozen` from `pyproject.toml` + `uv.lock`, copy code only. No models, no data.
   - [ ] `.dockerignore`: `data/`, `models/`, `classification/`, `output/`, `bad_samples/`, `samples/`, `.venv`, `.git`, `graphify-out/`, `.code-review-graph/`.
   - [ ] Check the image keeps the pinned stack: torch cu126 wheels and `onnxruntime-gpu` 1.22–1.23. Confirm both see the GPU (`nvidia-smi`, `ort.get_available_providers()`).
   - [ ] Run the container locally with `--gpus all`, mounting models and data, and pass the golden test.
6. **Models**
   - [ ] Write `models/manifest.yaml` listing every model file with SHA-256, size, source and version (replaces the hardcoded `copy_inference_files.sh`).
   - [ ] Add a `fetch_models` step (local path now, `gs://` later) that verifies checksums.
   - [ ] Decide whether `classification/*.pth` is needed at runtime (inference uses the ONNX). If not, leave it out of the runtime set.
7. **Repo hygiene**
   - [ ] Confirm large files (`*.pt`, `*.onnx`, `*.pth`, images) are gitignored. Record where each lives.
   - [ ] Remove or archive `copy_inference_files.sh` once the manifest exists.
   - [ ] Add a CI job (GitHub Actions or Cloud Build later): lint, unit tests, golden test on CPU if feasible, image build.
   - [ ] Keep Python work in the uv venv at `/projects/mosquitto_msis/code/.venv`.

**Exit gate:** the container built locally reproduces the golden outputs, and nothing in the code reads a workstation-specific path.

### Phase 2 — GCP foundation (1 day)
- [ ] Create the GCP project (separate dev and prod projects if budget allows). Link billing.
- [ ] Set a **budget with alerts** (50 / 90 / 100 %) before enabling GPUs.
- [ ] Enable APIs: Compute Engine, Cloud Storage, Artifact Registry, Cloud Build, Cloud Logging, plus Cloud Run and Vertex AI if chosen.
- [ ] IAM: one service account per role (job runner, build, data uploader), least privilege, **no downloaded JSON keys**. Use workload identity or attached service accounts.
- [ ] Buckets (single region matching compute), uniform bucket-level access, no public access:
  - `…-models` (versioning on)
  - `…-data-raw` (archive classes through lifecycle rules)
  - `…-outputs` (lifecycle: delete or cool down after N days)
- [ ] Artifact Registry Docker repo in the same region.
- [ ] **Request GPU quota now** (L4 or the chosen GPU, in the chosen region). Approval can take days, so do not leave it to the end.
- [ ] Keep the setup as a `gcloud` script in `docker/` or `infra/`, committed to the repo.

### Phase 3 — Move artifacts and data (1–3 days, mostly transfer time)
- [ ] Push the image to Artifact Registry (tag with git SHA, not only `latest`).
- [ ] Upload models to `gs://…-models/<version>/` and verify the manifest checksums.
- [ ] Upload only the data selected in D4 with `gcloud storage rsync` (resumable, parallel). Check the transfer time for the connection speed: at 100 MB/s, 245 GB takes about 40 minutes at best; at 10 MB/s, about 7 hours.
- [ ] Verify counts and checksums against a local manifest. Keep the manifest in git.
- [ ] Do not upload `*.zip` duplicates (e.g. `[Pharmcle] Mosquito.zip` duplicates the extracted folder).
- [ ] Choose the storage class by access pattern: Standard for active data, Nearline/Coldline/Archive for raw archives.

### Phase 4 — First cloud run, lift-and-shift (1–2 days)
- [ ] Create a GCE VM with an L4 GPU, Container-Optimized OS or an NVIDIA driver image, and an attached service account.
- [ ] Confirm the driver supports CUDA 12.x for the pinned wheels.
- [ ] Pull the image, run it against `gs://` input and output, run the golden set.
- [ ] Compare against the local baseline: results (within tolerance), images/sec, GPU memory, wall time, cost per 1,000 images.
- [ ] Use a Spot VM for the batch run if reruns are safe (they are, once outputs are idempotent).
- [ ] Stop or delete the VM after each run. A forgotten GPU VM is the most common surprise bill.

**Exit gate:** golden parity in the cloud and a cost-per-1,000-images number.

### Phase 5 — Managed and automated (optional, 2–5 days)
- [ ] Move batch to **Cloud Run Jobs with GPU** or **Vertex AI custom jobs** (check current GPU support, region and quota limits). Run the same image unchanged.
- [ ] Trigger: Cloud Scheduler for periodic runs, or Eventarc/Workflows on upload to `…-data-raw`.
- [ ] CI/CD: Cloud Build or GitHub Actions builds on push, runs the golden test, pushes the tagged image.
- [ ] Optional: load per-run metrics (counts per species, confidence, timing) into BigQuery for dashboards.
- [ ] Optional: training on Vertex AI, once the mmpretrain/ultralytics training setup is brought into a repo and containerized.

### Phase 6 — Operate and cut over (1–2 days)
- [ ] Alerts: job failure, runtime over threshold, budget overrun.
- [ ] Logs: confirm structured logs show up in Cloud Logging with run ID and image tag.
- [ ] Runbook (`docs/runbook.md`): how to run, rerun, roll back to the previous image/model version, rotate access.
- [ ] Rollback test: deploy the previous image and model version and confirm the golden set still passes.
- [ ] Run cloud and local in parallel for one cycle, compare results, then switch.
- [ ] Decide what happens to local data: keep as backup until the cloud copy is verified, then archive.

## 5. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| GPU quota or regional availability delays | Schedule slips | Request quota in Phase 2. Have a second region ready |
| CUDA / driver / `onnxruntime-gpu` mismatch in the image | Falls back to CPU silently or fails | Pin versions (already pinned). Check providers at startup and fail loudly if CUDA is missing |
| Results drift from local (precision, library versions) | Wrong classifications | Golden set with tolerance, run on every build |
| Vendor-SDK re-implementation is not bit-exact | Cloud and original app disagree | Treat local `infer.py` as the reference. Do not "fix" it during migration |
| Confidential customer data in a public cloud | Contract or legal breach | Resolve D6 before the first upload |
| 245 GB transfer slow or interrupted | Delays | Resumable `rsync`, upload only what is needed, run overnight |
| Forgotten GPU VM or job | Surprise cost | Budgets and alerts, auto-stop, Spot VMs, delete after runs |
| Models in git or the image | Slow builds, bloated registry | Models live in GCS with a checksum manifest |
| Credentials leaking | Security incident | Attached service accounts, no JSON keys, no secrets in the image or repo |

## 6. Master checklist

**Before touching GCP**
- [ ] D1–D7 decided and written down
- [ ] Data inventory done, confidentiality confirmed (D6)
- [ ] Local baseline recorded (speed, memory, outputs)
- [ ] Golden set + parity test passing
- [ ] Paths and config overridable by env/CLI
- [ ] Storage module handles local and `gs://`
- [ ] Structured logging, exit codes, idempotent reruns
- [ ] Dockerfile + `.dockerignore`, GPU verified in the container
- [ ] Model manifest with checksums; hardcoded drive paths removed
- [ ] CI runs tests and builds the image

**GCP foundation**
- [ ] Project, billing, budget alerts
- [ ] APIs enabled, service accounts least-privilege, no key files
- [ ] Buckets (models / data-raw / outputs) with lifecycle rules, no public access
- [ ] Artifact Registry repo
- [ ] GPU quota approved
- [ ] Setup scripted in the repo

**Move and run**
- [ ] Image pushed with a git-SHA tag
- [ ] Models uploaded and checksum-verified
- [ ] Required data uploaded and verified against a manifest
- [ ] First GPU run passes the golden set
- [ ] Cost per 1,000 images and throughput recorded
- [ ] VM/job auto-stop confirmed

**Operate**
- [ ] Alerts for failure and budget
- [ ] Runbook written, rollback tested
- [ ] Parallel run with local, results compared
- [ ] Cutover decision recorded; local data archived

## 7. Definition of done

The same container image produces golden-set-identical results locally and on GCP. Models and data are versioned in GCS with checksums. A run costs a known amount per 1,000 images. Budgets and alerts exist. Someone other than the author can run and roll back the pipeline using only the runbook.

## 8. Suggested order for the next two weeks

| Days | Work |
|---|---|
| 1 | Phase 0, decisions, data inventory, baseline |
| 2–4 | Phase 1: golden set, env/config, storage module, Dockerfile |
| 5 | Phase 1 wrap-up: manifest, CI, local container parity |
| 6 | Phase 2: GCP project, budget, buckets, registry, GPU quota request |
| 7–8 | Phase 3: upload image, models, selected data |
| 9–10 | Phase 4: GPU VM run, parity and cost numbers |
| After | Phase 5/6 based on D2 and D3 |
