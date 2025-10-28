import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image

_pipeline = None
_input_name: Optional[str] = None


def _ensure_pipeline():
    global _pipeline, _input_name
    if _pipeline is not None:
        return
    from bioimageio.core import load_resource, create_prediction_pipeline

    # Model location: prefer explicit path, else URL, else raise
    model_path = os.getenv("MITONET_MODEL_PATH")
    model_url = os.getenv("MITONET_MODEL_URL")
    if not model_path and not model_url:
        raise RuntimeError(
            "MitoNet model not configured. Set MITONET_MODEL_PATH to a local bioimage.io model folder/zip, "
            "or MITONET_MODEL_URL to a downloadable model resource."
        )
    src = model_path or model_url
    rd = load_resource(src)
    _pipeline = create_prediction_pipeline(rd)
    # find first input tensor name
    try:
        _input_name = rd.inputs[0].name  # type: ignore[attr-defined]
    except Exception:
        _input_name = "image"


def predict_mask_mitonet(pil_im: Image.Image, thr: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Run MitoNet via bioimage.io prediction pipeline.
    Returns (grayscale_array, mask_uint8).
    """
    _ensure_pipeline()
    assert _pipeline is not None
    assert _input_name is not None

    arr = np.array(pil_im.convert("L"))
    # Normalize to [0,1]
    x = arr.astype(np.float32) / 255.0
    # Pipeline expects channel/shape handling; try both HxW and 1xHxW
    sample = {_input_name: x}
    try:
        pred = _pipeline(sample)
    except Exception:
        sample = {_input_name: x[None, ...]}
        pred = _pipeline(sample)
    # Get first output as probability map
    try:
        first_key = list(pred.keys())[0]
        prob = np.array(pred[first_key])
        # squeeze any leading singleton dims
        while prob.ndim > 2:
            prob = prob.squeeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to run MitoNet pipeline: {e}")
    mask = (prob >= float(thr)).astype(np.uint8) * 255
    return arr, mask


