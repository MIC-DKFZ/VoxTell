"""FastAPI application exposing VoxTell inference over HTTP.

Designed for the single-user "GUI on a laptop, model on a workstation" split (see
``voxtell/server/__main__.py``). The flow is:

    POST /images         upload a .nii.gz once; server reorients to RAS + caches it
    POST /jobs           start a background segmentation for {image_id, prompts}
    GET  /jobs/{id}/events   stream NDJSON per-patch progress until a terminal status
    GET  /jobs/{id}/result   download the masks (blosc2) once the job is done
    POST /jobs/{id}/cancel   cooperatively cancel a running job
    POST /images/{id}/export write the labelmap back in the original orientation

Reorientation is done with nnU-Net's ``NibabelIOWithReorient`` (the exact training
reader) so the server is the single source of truth for orientation; the client just
displays the reoriented array it gets back and overlays masks in the same space.

The client/server architecture and the blosc2 wire format are inspired by MIC-DKFZ's
nnInteractive / napari-nninteractive (Apache-2.0).
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

from voxtell.inference.predictor import InferenceCancelled, VoxTellPredictor
from voxtell.server.runner import RemoteInferenceEngine
from voxtell.server.serialization import pack_array, unpack_array

# JSON metadata header carried alongside a binary (blosc2) array body.
META_HEADER = "X-Meta"
CONTENT_TYPE_OCTET_STREAM = "application/octet-stream"

# Reap cached images/jobs untouched for this long (seconds); single-user, generous.
_DEFAULT_IDLE_TIMEOUT = 3600


@dataclass
class _Image:
    """A reoriented image cached server-side, keyed by image_id."""

    data: np.ndarray  # (1, Z, Y, X), RAS, ready for the predictor
    props: dict  # NibabelIOWithReorient props (for the inverse reorientation on export)
    spacing: tuple
    last_used: float


@dataclass
class _Job:
    """A background segmentation and its live progress/result state."""

    image_id: str
    prompts: List[str]
    keep_largest: bool
    status: str = "running"  # running | done | cancelled | error
    result: Optional[np.ndarray] = None
    error: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    events: "queue.Queue" = field(default_factory=queue.Queue)
    created: float = field(default_factory=time.monotonic)


def make_app(
    predictor: VoxTellPredictor,
    model_name: str = "voxtell",
    api_key: Optional[str] = None,
    idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
):
    """Build the FastAPI app around an already-constructed ``predictor``."""
    engine = RemoteInferenceEngine(predictor)
    images: Dict[str, _Image] = {}
    jobs: Dict[str, _Job] = {}
    lock = threading.Lock()

    def require_auth(authorization: Optional[str] = Header(default=None)):
        """Optional static bearer-token gate (no-op when no api_key is configured)."""
        if api_key is None:
            return
        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    app = FastAPI(title="VoxTell inference server")

    def _touch(image_id: str):
        img = images.get(image_id)
        if img is not None:
            img.last_used = time.monotonic()

    def _reap():
        """Drop images and finished jobs that have been idle past the timeout."""
        now = time.monotonic()
        with lock:
            for image_id in [k for k, v in images.items() if now - v.last_used > idle_timeout]:
                images.pop(image_id, None)
            for job_id in [
                k for k, v in jobs.items()
                if v.status != "running" and now - v.created > idle_timeout
            ]:
                jobs.pop(job_id, None)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/capabilities", dependencies=[Depends(require_auth)])
    def capabilities():
        return {
            "model": model_name,
            "patch_size": list(predictor.patch_size),
            "available_embeddings": predictor.list_available_embeddings(),
        }

    @app.post("/images", dependencies=[Depends(require_auth)])
    async def upload_image(request: Request, include_image: bool = True):
        """Accept raw .nii.gz bytes, reorient to RAS, cache, and return metadata.

        When ``include_image`` is true the reoriented (Z, Y, X) array is returned in the
        body for display; the GUI sets this false when it already loaded the image
        locally with the same reader and only needs the ``image_id``.
        """
        raw = await request.body()
        _reap()
        # NibabelIOWithReorient reads from a path and infers the format from the suffix.
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            data, props = NibabelIOWithReorient().read_images([tmp_path])
        finally:
            os.unlink(tmp_path)

        image_id = uuid.uuid4().hex
        spacing = tuple(float(s) for s in props["spacing"])
        with lock:
            images[image_id] = _Image(
                data=data, props=props, spacing=spacing, last_used=time.monotonic()
            )

        meta = {
            "image_id": image_id,
            "spacing": list(spacing),
            "shape": list(data.shape[1:]),
        }
        # Body = the reoriented (Z, Y, X) array for display (masks come back in this
        # space), or empty when the GUI already holds the locally-reoriented image.
        body = pack_array(np.ascontiguousarray(data[0])) if include_image else b""
        return Response(
            content=body,
            media_type=CONTENT_TYPE_OCTET_STREAM,
            headers={META_HEADER: json.dumps(meta)},
        )

    @app.post("/jobs", dependencies=[Depends(require_auth)])
    def start_job(payload: dict):
        image_id = payload.get("image_id")
        prompts = payload.get("prompts") or []
        keep_largest = bool(payload.get("keep_largest", False))
        with lock:
            image = images.get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Unknown image_id (upload it first).")
        if not prompts:
            raise HTTPException(status_code=400, detail="No prompts given.")
        _touch(image_id)

        job = _Job(image_id=image_id, prompts=list(prompts), keep_largest=keep_largest)
        job_id = uuid.uuid4().hex
        with lock:
            jobs[job_id] = job
        threading.Thread(target=_run_job, args=(job, image), daemon=True).start()
        return {"job_id": job_id}

    def _run_job(job: _Job, image: _Image):
        def progress(done: int, total: int) -> bool:
            job.events.put({"done": done, "total": total})
            return not job.cancel.is_set()

        def notice(message: str):
            job.events.put({"event": "oom_fallback", "message": message})

        try:
            masks = engine.segment(
                image.data, job.prompts, job.keep_largest,
                progress_callback=progress, notice_callback=notice,
            )
            job.result = masks
            job.status = "done"
            job.events.put({"status": "done"})
        except InferenceCancelled:
            job.status = "cancelled"
            job.events.put({"status": "cancelled"})
        except Exception as exc:  # noqa: BLE001 - report any failure to the client
            job.status = "error"
            job.error = str(exc)
            job.events.put({"status": "error", "message": str(exc)})

    @app.get("/jobs/{job_id}/events", dependencies=[Depends(require_auth)])
    def stream_events(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id.")

        def generate():
            while True:
                event = job.events.get()
                yield json.dumps(event) + "\n"
                if "status" in event:  # terminal event ends the stream
                    break

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.get("/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
    def job_result(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id.")
        if job.status == "error":
            raise HTTPException(status_code=500, detail=job.error or "Inference failed.")
        if job.status != "done" or job.result is None:
            raise HTTPException(status_code=409, detail=f"Job not finished (status={job.status}).")
        _touch(job.image_id)
        return Response(
            content=pack_array(job.result), media_type=CONTENT_TYPE_OCTET_STREAM
        )

    @app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
    def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id.")
        job.cancel.set()
        return {"cancelling": True}

    @app.post("/images/{image_id}/export", dependencies=[Depends(require_auth)])
    async def export_labelmap(image_id: str, request: Request):
        """Write a labelmap back in the image's original orientation, return .nii.gz bytes."""
        with lock:
            image = images.get(image_id)
        if image is None:
            raise HTTPException(status_code=404, detail="Unknown image_id.")
        labelmap = unpack_array(await request.body())
        _touch(image_id)
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            out_path = tmp.name
        try:
            NibabelIOWithReorient().write_seg(labelmap, out_path, image.props)
            with open(out_path, "rb") as handle:
                content = handle.read()
        finally:
            os.unlink(out_path)
        return Response(content=content, media_type=CONTENT_TYPE_OCTET_STREAM)

    return app
