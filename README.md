# mito-tracker

## Overview
MitoTracker is a web app for segmenting mitochondria in electron micrographs (EMs). It combines a high-precision segmentation backend (MitoNet via BioImage.IO, with a classical fallback) and a modern Next.js frontend that visualizes overlays, metrics, and optional LLM analysis.

### Goals
- Prioritize precision over recall: only identify mitochondria similar to training data.
- Enforce biological cues on uploads: double membrane, cristae, circularity, darkness vs background, and minimum size.
- Avoid whitespace and text-like artifacts via morphology and shape filters.

## Repository layout
- `backend/`: FastAPI service for inference and metrics
- `frontend/`: Next.js app for UI/UX
- `data/`: Example train/val layout (images, masks)
- `models/`: Local model assets (keep large weights out of Git)
- `scripts/`: Utilities (e.g., pseudo-labeling)

## Backend
Stack: FastAPI, Pillow, OpenCV, scikit-image.

Endpoints:
- `GET /health`: Health probe
- `POST /seg`: Segmentation for an uploaded file or curated example

### Segmentation flow (`/seg`)
1) Input
- Uploaded image: binary file
- Curated example: `example_id` to use server-side images and masks

2) Model inference
- Default: MitoNet (BioImage.IO) using `MITONET_MODEL_PATH` or `MITONET_MODEL_URL`
- Fallback: classical segmentation (adaptive threshold + morphology) if MitoNet errors

3) Optional LLM assist
- If `use_llm=true` and `OPENAI_API_KEY` is set, the server prompts a vision model to return polygons (JSON only)
- Fusion: uploads use intersection (stricter), examples use union

4) Example shaping (for curated examples)
- Bypass strict biological gates
- Confidence-driven dilate/erode, optional watershed splitting, and top‑K subsampling so the confidence slider visibly affects counts

5) Strict gates (uploads only)
- Circularity threshold, double-membrane edges via Canny in inner/outer rings, cristae via Laplacian energy, darkness vs local background, and scale-aware minimum area

6) Outputs
- Overlay PNG (original EM + translucent color), mask PNG, and per-component metrics
- When `pixel_size_nm` is provided: `area_um2`, `perimeter_um`, `length_um`, `width_um`

### Environment variables
- `MITONET_MODEL_PATH`: Local BioImage.IO model folder/zip (preferred)
- `MITONET_MODEL_URL`: Remote BioImage.IO resource if PATH not set
- `OPENAI_API_KEY`: Enables LLM polygon extraction and 2–3 sentence analysis
- `NEXT_PUBLIC_API_URL`: Frontend → backend base URL

## Local development
### Backend
- Create venv and install deps:
  - `pip install -r backend/requirements.txt`
- Set a MitoNet source:
  - `export MITONET_MODEL_PATH="/abs/path/to/mitonet_model"` (preferred)
  - or `export MITONET_MODEL_URL="<bioimage.io identifier or URL>"`
- Run API:
  - `uvicorn backend.main:app --host 127.0.0.1 --port 8000`

### Frontend
- Node 18+
- `cd frontend && npm install && npm run dev`

## Usage
- Choose an example (1–4) to see curated behavior; the slider tunes mask size/count
- Upload your own EM and set pixel size (nm) for scale-aware metrics
- Toggle “Use LLM labels” to fuse LLM polygons with model predictions
- The overlay card shows colored segmentation; the metrics card includes counts and geometry

## Training
Two paths are supported.

### 1) UNet (local training)
- 3-channel inputs (intensity, Sobel magnitude, Laplacian-of-Gaussian)
- Loss: BCE + Dice + boundary Dice to emphasize borders
- Augmentations: size-consistent transforms, relaxed shape checks, and pair filtering
- Data layout: `data/train|val/images` and `data/train|val/masks`
- Pseudo-labeling: `scripts/pseudo_label.py` to augment with model-generated labels
- Note: runtime inference uses MitoNet for higher precision; UNet remains for experiments

### 2) MitoNet (preferred)
- Loaded via BioImage.IO; generally outperforms basic UNet baselines on EM mitochondria
- For fine-tuning, prepare a consistent dataset (e.g., Empanada subset), train per upstream guidance, then export a BioImage.IO bundle to use as `MITONET_MODEL_PATH`

## Design principles for precision
- Require consistent double-membrane edges
- Prefer internal cristae-like texture over smooth blobs
- Avoid bright regions and watermark/text components
- Prefer missing a few over marking non-mito structures

## Troubleshooting
- “Network error” on upload: backend now falls back to classical segmentation on model failure and returns 200
- Examples show no metrics: ensure `example_id` is set and example assets exist under `backend/examples/<id>/`
- LLM assist missing: set `OPENAI_API_KEY` and verify model access

## Production deployment (permanent backend URL)
Use Render to host the FastAPI backend so the site always works (no tunnels).

### One‑time setup
1. Push this repo to GitHub (if not already).
2. Go to Render → New → Blueprint → select your repo.
3. Render will detect `render.yaml` at the repo root. Accept defaults.
   - Service type: Web (Python)
   - Root directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Env vars: `MITONET_MODEL_URL=stupendous-sheep/1.1`. Add `OPENAI_API_KEY` if using LLM features.
4. Click Deploy. When live, copy the service URL (e.g., `https://mito-tracker-backend.onrender.com`).

### Wire frontend to backend
1. In Vercel → Project → Settings → Environment Variables:
   - Key: `BACKEND_URL`
   - Value: your Render URL (e.g., `https://mito-tracker-backend.onrender.com`)
   - Scope: All Environments
2. Redeploy the site. The frontend proxies `/api/seg` and example images to `BACKEND_URL`.

Notes
- Add curated examples at `backend/examples/<id>/image.png` and `mask.png` to enable the example selector.
- If you later fine‑tune MitoNet, set `MITONET_MODEL_PATH` to your BioImage.IO bundle and remove `MITONET_MODEL_URL`.

## License
MIT