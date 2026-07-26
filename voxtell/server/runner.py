"""Server-side inference engine wrapping :class:`VoxTellPredictor`.

This holds the GPU-side logic that used to live in the napari plugin's
``ProcessingThread``: batching prompts to fit VRAM (with graceful OOM fallback to a
smaller batch, then to CPU accumulation) and the optional keep-largest-component
post-processing. Moving it here keeps the laptop client torch-free — it only sends
prompts and receives masks.
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

import numpy as np
import torch

from voxtell.inference.predictor import InferenceCancelled, VoxTellPredictor

# Callback signatures shared with the HTTP layer.
ProgressCallback = Callable[[int, int], bool]  # (done, total) -> keep going?
NoticeCallback = Callable[[str], None]  # human-readable status (e.g. OOM fallback)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary mask to its single largest connected component."""
    from scipy.ndimage import label as cc_label

    labeled, n = cc_label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    return (labeled == sizes.argmax()).astype(mask.dtype)


class RemoteInferenceEngine:
    """Runs one segmentation on a shared predictor, batching prompts to fit memory."""

    def __init__(self, predictor: VoxTellPredictor):
        self.predictor = predictor
        # Serialize all GPU work: the predictor is shared and mutable
        # (``perform_everything_on_device``), so overlapping jobs would race on it and
        # contend for VRAM. Queued jobs simply wait their turn.
        self._lock = threading.Lock()

    def segment(
        self,
        image_data: np.ndarray,
        prompts: List[str],
        keep_largest: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
        notice_callback: Optional[NoticeCallback] = None,
    ) -> np.ndarray:
        """Segment ``image_data`` (RAS, ``(1, Z, Y, X)`` or ``(Z, Y, X)``) with ``prompts``.

        Returns masks of shape ``(num_prompts, Z, Y, X)`` as ``uint8``. Serialized across
        concurrent callers by an internal lock.
        """
        with self._lock:
            masks = self._predict_with_oom_fallbacks(
                image_data, prompts, progress_callback, notice_callback
            )
        if keep_largest:
            masks = np.stack([_keep_largest_component(m) for m in masks])
        return masks

    def _predict(self, image_data, prompts, on_device, progress_callback) -> np.ndarray:
        """One predictor call with the logits accumulator on GPU (fast) or CPU (safe)."""
        self.predictor.perform_everything_on_device = on_device
        return self.predictor.predict_single_image(
            image_data, prompts, progress_callback=progress_callback
        ).astype(np.uint8)

    @staticmethod
    def _is_oom(err) -> bool:
        return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()

    def _free_gpu(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initial_batch_size(self, image_data: np.ndarray, n_prompts: int) -> int:
        """Largest prompt batch likely to fit in the currently free GPU memory.

        The per-prompt logits accumulator (~volume x 2 bytes) dominates, so we size the
        batch from free VRAM up-front instead of trying all prompts and OOM-ing slowly.
        """
        device = getattr(self.predictor, "device", None)
        if not (torch.cuda.is_available() and getattr(device, "type", "") == "cuda"):
            return n_prompts
        try:
            free, _ = torch.cuda.mem_get_info()
        except Exception:  # noqa: BLE001 - any failure -> just try all prompts at once
            return n_prompts
        voxels = int(np.prod(image_data.shape[-3:]))
        per_prompt = voxels * 2  # half-precision logits accumulator, bytes
        # Reserve ~60% of free memory for the network, activations and n_predictions.
        fit = int(free * 0.4 / max(per_prompt, 1))
        return max(1, min(n_prompts, fit))

    def _predict_with_oom_fallbacks(
        self,
        image_data: np.ndarray,
        prompts: List[str],
        progress_callback: Optional[ProgressCallback],
        notice_callback: Optional[NoticeCallback],
    ) -> np.ndarray:
        """Predict in GPU batches sized to fit memory, degrading gracefully on OOM."""
        def notify(message: str):
            if notice_callback is not None:
                notice_callback(message)

        original = getattr(self.predictor, "perform_everything_on_device", True)
        try:
            prompts = list(prompts)
            size = self._initial_batch_size(image_data, len(prompts))
            if size < len(prompts):
                notify(f"Segmenting in batches of {size} prompt(s) to fit GPU memory...")
            masks, i = [], 0
            while i < len(prompts):
                sub = prompts[i:i + size]
                try:
                    masks.append(self._predict(image_data, sub, True, progress_callback))
                    i += size
                except InferenceCancelled:
                    raise
                except RuntimeError as e:
                    if not self._is_oom(e):
                        raise
                    self._free_gpu()
                    if size > 1:
                        size = max(1, size // 2)
                        notify(f"Low GPU memory - reducing batch to {size}...")
                        continue
                    notify(
                        "A prompt is too large for GPU memory - using CPU accumulation (slow)..."
                    )
                    masks.append(self._predict(image_data, sub, False, progress_callback))
                    i += 1
            return masks[0] if len(masks) == 1 else np.concatenate(masks, axis=0)
        finally:
            self.predictor.perform_everything_on_device = original
