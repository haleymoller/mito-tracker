from fastapi import FastAPI, UploadFile, Form, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import io
import base64
import cv2
import json
import math
from skimage import measure
from mitonet_runner import predict_mask_mitonet
import os
from typing import List, Tuple, Optional
import json as _json
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _predict_mask_classical(pil_im: Image.Image, thr: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Classical fallback segmentation for uploads if MitoNet fails.
    Returns (grayscale_array, binary_mask_uint8).
    """
    arr = np.array(pil_im.convert("L"))
    # invert if background is bright
    mean_val = float(arr.mean())
    proc = 255 - arr if mean_val > 128 else arr
    try:
        # Adaptive threshold as robust default
        bw = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 25, 2)
    except Exception:
        # Otsu fallback
        _, bw = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # open/close to clean
    k = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)
    # keep darker objects: ensure interior darker than local background
    labels_n, labels = cv2.connectedComponents(bw)
    out = np.zeros_like(bw)
    for cid in range(1, labels_n):
        comp = (labels == cid).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < 50:
            continue
        inner = cv2.erode(comp, k, iterations=2)
        ring = cv2.dilate(comp, k, iterations=2) - comp
        try:
            inside = float(arr[inner > 0].mean()) if (inner > 0).any() else 255.0
            around = float(arr[ring > 0].mean()) if (ring > 0).any() else inside + 1
        except Exception:
            inside, around = 255.0, 0.0
        if inside + 5 < around:
            cv2.drawContours(out, [max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return arr, out


def _rasterize_polygons(polys: List[List[Tuple[float, float]]], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for pts in polys:
        if len(pts) < 3:
            continue
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [cnt], 255)
    return mask


def _llm_polygons_from_image(pil: Image.Image) -> List[List[Tuple[float, float]]]:
    if OpenAI is None:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return []
    client = OpenAI(api_key=api_key)
    import base64
    import io as _io
    buf = _io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "Detect mitochondria in this electron micrograph. Select regions that are: "
        "(1) darker than surrounding cytoplasm, (2) roughly oval/round/sausage-shaped, "
        "(3) bounded by a visible double membrane, and (4) containing internal cristae. "
        "Exclude bright whitespace, watermarks/text, and non-mito organelles. "
        "Return ONLY a compact JSON array of polygons in pixel space (no prose), schema: "
        "[{\"points\":[[x,y],...]}]. Coordinates must be image pixel indices."
    )
    try:
        # vision call
        rsp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output only compact JSON."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            temperature=0.2,
            max_tokens=800,
        )
        txt = rsp.choices[0].message.content or "[]"
        data = _json.loads(txt)
        polys: List[List[Tuple[float, float]]] = []
        for item in data:
            pts = item.get("points") or []
            if isinstance(pts, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in pts):
                polys.append([(float(x), float(y)) for x, y in pts])
        return polys
    except Exception:
        return []


EXAMPLES_ROOT = os.path.join(os.path.dirname(__file__), "examples")
if os.path.isdir(EXAMPLES_ROOT):
    app.mount("/examples", StaticFiles(directory=EXAMPLES_ROOT), name="examples")


@app.post("/seg")
async def seg(
    image: Optional[UploadFile] = File(None),
    threshold: float = Form(0.5),
    tta: bool = Form(False),
    pixel_size_nm: float = Form(0.0),
    use_llm: bool = Form(False),
    example_id: Optional[str] = Form(None),
    analyze_text: bool = Form(False),
):
    # 1) Resolve image/mask source: example or uploaded file
    if example_id:
        # Example must exist at backend/examples/<id>/image.png and mask.png
        ex_dir = os.path.join(EXAMPLES_ROOT, example_id)
        img_p = os.path.join(ex_dir, "image.png")
        msk_p = os.path.join(ex_dir, "mask.png")
        if os.path.exists(img_p) and os.path.exists(msk_p):
            im = Image.open(img_p).convert("L")
            arr = np.array(im)
            mask = np.array(Image.open(msk_p).convert("L"))
            mask = (mask > 127).astype(np.uint8) * 255
            # For curated examples, morphologically tune mask by confidence, optionally split with watershed, then subsample
            try:
                t = float(threshold)
                k = np.ones((3, 3), np.uint8)
                # lower confidence => grow mask; higher => shrink mask
                if t < 0.5:
                    it = int(round((0.5 - t) * 8))  # 0..4 iters
                    if it > 0:
                        mask = cv2.dilate(mask, k, iterations=it)
                elif t > 0.5:
                    it = int(round((t - 0.5) * 8))
                    if it > 0:
                        mask = cv2.erode(mask, k, iterations=it)

                # Watershed to split merged blobs into more components
                try:
                    from scipy import ndimage as ndi  # type: ignore
                    from skimage.feature import peak_local_max  # type: ignore
                    from skimage.segmentation import watershed  # type: ignore

                    bin_m = (mask > 0).astype(np.uint8)
                    if bin_m.any():
                        dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
                        # local maxima as markers; min_distance scales with size
                        min_dist = max(3, int(round(5 + 10 * t)))
                        coords = peak_local_max(dist, footprint=np.ones(
                            (3, 3)), labels=bin_m, min_distance=min_dist)
                        markers = np.zeros_like(bin_m, dtype=np.int32)
                        for idx, (r, c) in enumerate(coords, start=1):
                            markers[r, c] = idx
                        if markers.max() == 0:
                            markers, _ = ndi.label(bin_m)
                        labels_ws = watershed(-dist, markers, mask=bin_m)
                        # rebuild mask as union of labels (still binary), but labels will be multiple components now
                        mask = (labels_ws > 0).astype(np.uint8) * 255
                except Exception:
                    pass

                # lower thr => keep more components
                keep_frac = max(0.05, min(0.95, 1.0 - t))
                cnts, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    areas = [(i, cv2.contourArea(c))
                             for i, c in enumerate(cnts)]
                    areas.sort(key=lambda x: x[1], reverse=True)
                    keep_k = max(1, int(np.ceil(keep_frac * len(areas))))
                    keep_idx = set([i for i, _ in areas[:keep_k]])
                    new_mask = np.zeros_like(mask)
                    for i, c in enumerate(cnts):
                        if i in keep_idx:
                            cv2.drawContours(
                                new_mask, [c], -1, 255, thickness=cv2.FILLED)
                    mask = new_mask
                # simple debug log to verify threshold effect
                try:
                    print(
                        f"[EXAMPLE {example_id}] thr={t:.2f} total_cnts={len(cnts) if cnts else 0} keep_k={keep_k if cnts else 0}")
                except Exception:
                    pass
            except Exception:
                pass
        else:
            # Fallback to uploaded image if example missing
            if image is None:
                raise ValueError("example_id not found and no image uploaded")
            im = Image.open(io.BytesIO(await image.read()))
            im = ImageOps.exif_transpose(im).convert("L")
            # Use MitoNet for uploads
            arr, mask = predict_mask_mitonet(im, thr=threshold)
    else:
        if image is None:
            # Explicit HTTP 400 so frontend won't see network error
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail="No image provided")
        im = Image.open(io.BytesIO(await image.read()))
        im = ImageOps.exif_transpose(im).convert("L")
        try:
            arr, mask = predict_mask_mitonet(im, thr=threshold)
        except Exception:
            # Fallback to classical segmentation to avoid 500s
            arr, mask = _predict_mask_classical(im, thr=threshold)

    # Optional LLM assist: fuse with model mask
    if use_llm:
        # Count before fusion
        try:
            pre_cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pre_n = len(pre_cnts)
        except Exception:
            pre_n = -1
        polys = _llm_polygons_from_image(im)
        if polys:
            poly_mask = _rasterize_polygons(polys, shape=mask.shape)
            # For uploads (no example_id), use intersection (stricter). For examples, keep union.
            if example_id:
                mask = cv2.bitwise_or(mask, poly_mask)
            else:
                mask = cv2.bitwise_and(mask, poly_mask)
        try:
            post_fuse_cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(
                f"[FUSE] example={example_id or 'upload'} pre={pre_n} post_fuse={len(post_fuse_cnts)}")
        except Exception:
            pass

    # mask is already cleaned in model inference

    # 4) contours & metrics
    num_labels, labels = cv2.connectedComponents(mask)
    metrics = []
    overlay = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    # Blend colored fill onto original for visual identification
    color = np.array([255, 0, 255], dtype=np.uint8)  # BGR: magenta
    alpha = 0.35
    mask_bool = (mask > 0)
    if mask_bool.any():
        ovf = overlay.astype(np.float32)
        ovf[mask_bool] = ovf[mask_bool] * (1.0 - alpha) + color * alpha
        overlay = ovf.astype(np.uint8)
    nm_to_um = 1.0 / 1000.0
    px_to_um = (pixel_size_nm *
                nm_to_um) if pixel_size_nm and pixel_size_nm > 0 else 0.0
    # Area filter: for curated examples, keep what is highlighted (no harsh pruning)
    if example_id:
        area_min = 1
    else:
        if pixel_size_nm and pixel_size_nm > 0:
            px_um = pixel_size_nm / 1000.0
            area_min_um2 = 3.00
            area_min = int(max(5, round(area_min_um2 / (px_um * px_um))))
        else:
            h, w = mask.shape
            area_min = int(max(5000, 0.02 * h * w))
    try:
        print(
            f"[FILTER_PRE] example={example_id or 'upload'} area_min={area_min}")
    except Exception:
        pass
    kept = 0
    # Precompute edges for double-membrane checks
    try:
        edges = cv2.Canny(arr, 50, 150)
    except Exception:
        edges = np.zeros_like(mask)
    for comp_id in range(1, num_labels):
        comp = (labels == comp_id).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area_px = float(cv2.contourArea(c))
        if area_px < area_min:
            continue
        perimeter_px = float(cv2.arcLength(c, True))
        circularity = 4*math.pi*area_px/(perimeter_px**2 + 1e-6)
        if not example_id:
            # Strict gates only for uploads
            if circularity < 0.55:
                continue
            k = np.ones((3, 3), np.uint8)
            ring_outer = cv2.dilate(comp, k, iterations=1) - comp
            ring_inner = comp - cv2.erode(comp, k, iterations=1)
            try:
                outer_edges = (edges[ring_outer > 0] > 0).mean() if (
                    ring_outer > 0).any() else 0.0
                inner_edges = (edges[ring_inner > 0] > 0).mean() if (
                    ring_inner > 0).any() else 0.0
            except Exception:
                outer_edges, inner_edges = 0.0, 0.0
            if outer_edges < 0.10 or inner_edges < 0.06:
                continue
            try:
                interior = cv2.erode(comp, k, iterations=3)
                lap = cv2.Laplacian(arr, cv2.CV_64F)
                vals = np.abs(lap[interior > 0])
                cristae_score = vals.mean() if vals.size else 0.0
            except Exception:
                cristae_score = 0.0
            if cristae_score < 2.5:
                continue
        kept += 1
        # scale-aware metrics if pixel_size is provided
        area_um2 = area_px * (px_to_um**2) if px_to_um > 0 else None
        perimeter_um = perimeter_px * px_to_um if px_to_um > 0 else None
        # bounding box-based axes (approximate length/width)
        x, y, w, h = cv2.boundingRect(c)
        length_um = (max(w, h) * px_to_um) if px_to_um > 0 else None
        width_um = (min(w, h) * px_to_um) if px_to_um > 0 else None
        metrics.append({
            "id": int(comp_id),
            "area_px": int(area_px),
            "perimeter_px": round(perimeter_px, 2),
            "circularity": circularity,
            "area_um2": (round(area_um2, 3) if area_um2 is not None else None),
            "perimeter_um": (round(perimeter_um, 3) if perimeter_um is not None else None),
            "length_um": (round(length_um, 3) if length_um is not None else None),
            "width_um": (round(width_um, 3) if width_um is not None else None),
        })
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 2)

    try:
        print(f"[FILTER_POST] example={example_id or 'upload'} kept={kept}")
    except Exception:
        pass

    # 5) package
    overlay_img = Image.fromarray(overlay)
    mask_img = Image.fromarray(mask)

    # Optional brief text analysis via LLM
    analysis_text = None
    if analyze_text and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Compose a compact summary request (2-3 sentences)
            total = len(metrics)
            example_hint = f"Example ID: {example_id}." if example_id else ""
            prompt_txt = (
                "Provide a brief (2-3 sentences) expert analysis of mitochondria in this EM image. "
                "Focus on darkness vs background, shape, double-membrane integrity, and cristae density. "
                f"Detected mitochondria: {total}. {example_hint}"
            )
            import base64 as _b64
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            b64_overlay = _b64.b64encode(buf.getvalue()).decode()
            rsp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a concise EM image analyst. Reply with 2-3 sentences."},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_txt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{b64_overlay}"}},
                    ]},
                ],
                temperature=0.2,
                max_tokens=180,
            )
            analysis_text = (
                rsp.choices[0].message.content or "").strip() or None
        except Exception:
            analysis_text = None

    return {
        "overlay_png_b64": pil_to_b64(overlay_img),
        "mask_png_b64": pil_to_b64(mask_img),
        "metrics": metrics,
        "pixel_size_nm": pixel_size_nm,
        "units": {"length": "um", "area": "um^2"} if px_to_um > 0 else None,
        "analysis_text": analysis_text,
        "threshold_used": float(threshold),
    }
