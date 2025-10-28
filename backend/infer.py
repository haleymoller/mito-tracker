from model.train_unet import UNet  # reuse definition
import os
import sys
import math
from PIL import Image
import cv2
import numpy as np
import torch
from skimage.filters import threshold_otsu

# Ensure repo root is on sys.path before importing UNet
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


device = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if hasattr(torch.backends, "mps")
     and torch.backends.mps.is_available() else "cpu")
)
net = UNet(in_channels=3).to(device)

# Allow unet_best.pt to be in project root, backend/, or model/
CHECKPOINT_CANDIDATES = [
    os.path.join(REPO_ROOT, "unet_best.pt"),
    os.path.join(os.path.dirname(__file__), "unet_best.pt"),
    os.path.join(REPO_ROOT, "model", "unet_best.pt"),
]
ckpt_path = next((p for p in CHECKPOINT_CANDIDATES if os.path.exists(p)), None)
try:
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        target = net.state_dict()
        filtered = {k: v for k, v in state.items(
        ) if k in target and target[k].shape == v.shape}
        missing = set(target.keys()) - set(filtered.keys())
        unexpected = set(state.keys()) - set(filtered.keys())
        if filtered:
            net.load_state_dict(filtered, strict=False)
        # Always eval mode
        net.eval()
        if missing or unexpected:
            print(
                f"[infer] UNet checkpoint loaded partially. missing={len(missing)} unexpected={len(unexpected)}")
    else:
        net.eval()
except Exception as e:
    print(f"[infer] Skipping checkpoint load due to error: {e}")
    net.eval()


def _pad_to_multiple(arr: np.ndarray, multiple: int = 8) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = arr.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0, 0, 0)
    arr_p = np.pad(arr, ((top, bottom), (left, right)), mode="edge")
    return arr_p, (top, bottom, left, right)


def _unpad(arr: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    if top == bottom == left == right == 0:
        return arr
    h, w = arr.shape
    return arr[top:h-bottom if bottom > 0 else h, left:w-right if right > 0 else w]


def _make_3ch_features(gray: np.ndarray) -> np.ndarray:
    im_f = gray.astype(np.float32)
    sobelx = cv2.Sobel(im_f, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(im_f, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    if sobel_mag.max() > 0:
        sobel_mag = sobel_mag / (sobel_mag.max()+1e-6) * 255.0
    blur = cv2.GaussianBlur(im_f, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)
    if lap_abs.max() > 0:
        lap_abs = lap_abs / (lap_abs.max()+1e-6) * 255.0
    im3 = np.stack([im_f, sobel_mag, lap_abs], axis=-1).astype(np.uint8)
    return im3


def _predict_logits(numpy_gray: np.ndarray, max_side: int | None = 1024) -> np.ndarray:
    # Optional downscale for speed if image is large
    h0, w0 = numpy_gray.shape
    if max_side is not None:
        max_dim = max(h0, w0)
        if max_dim > max_side:
            scale = max_side / max_dim
            nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
            numpy_gray = cv2.resize(
                numpy_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    # Pad to multiple of 8 for UNet skip compatibility
    padded, pads = _pad_to_multiple(numpy_gray, multiple=8)
    feats = _make_3ch_features(padded)
    im_n = (feats/255.0 - 0.5)/0.5  # [H,W,3]
    x = torch.from_numpy(im_n).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = net(x)[0, 0].detach().cpu().numpy()
    logits = _unpad(logits, pads)
    # Resize back to original if we downscaled
    if logits.shape != (h0, w0):
        logits = cv2.resize(logits, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return logits


def predict_mask(pil_im: Image.Image, thr: float = 0.5, tta: bool = False, max_side: int | None = 1024):
    im = np.array(pil_im.convert("L"))
    if not tta:
        logits = _predict_logits(im, max_side=max_side)
        p = 1/(1+np.exp(-logits))
    else:
        # simple TTA: identity, hflip, vflip, hvflip
        flips = [
            (lambda a: a, lambda a: a),
            (np.fliplr, np.fliplr),
            (np.flipud, np.flipud),
            (lambda a: np.flipud(np.fliplr(a)), lambda a: np.fliplr(np.flipud(a)))
        ]
        preds = []
        for fwd, inv in flips:
            logits = _predict_logits(fwd(im), max_side=max_side)
            preds.append(inv(1/(1+np.exp(-logits))))
        p = np.mean(preds, axis=0)
    # initial threshold
    mask = (p >= thr).astype(np.uint8)*255
    # adaptive fallback if too few positives
    if mask.sum() < 100:  # very sparse
        try:
            auto_thr = float(threshold_otsu(p))
            mask = (p >= auto_thr).astype(np.uint8)*255
        except Exception:
            pass
    # basic clean
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # suppress very bright regions (likely whitespace) and thin text-like components
    # Bright: remove pixels where original is very light
    bright = (im >= 240).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright))
    # Remove thin elongated components by aspect ratio / skeleton-like width
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for comp_id in range(1, num_labels):
        x, y, w, h, area = stats[comp_id]
        if area < 15:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar > 12 and area < 500:
            # likely text stroke
            continue
        # Darkness gate: component interior must be darker than local ring by a margin
        comp_mask = (labels == comp_id).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(comp_mask, kernel, iterations=2)
        ring = cv2.subtract(dil, comp_mask)
        inside_vals = im[comp_mask.astype(bool)]
        ring_vals = im[ring.astype(bool)]
        if inside_vals.size == 0 or ring_vals.size == 0:
            continue
        mean_inside = float(np.mean(inside_vals))
        mean_ring = float(np.mean(ring_vals))
        if (mean_ring - mean_inside) < 5.0:  # require component to be at least ~5 gray levels darker
            continue
        cleaned[labels == comp_id] = 255
    mask = cleaned
    return im, mask
