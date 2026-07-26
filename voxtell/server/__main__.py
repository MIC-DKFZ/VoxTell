#!/usr/bin/env python3
"""``voxtell-server``: serve VoxTell inference over HTTP for the napari remote client.

Run this on the machine with the GPU/model; drive it from a laptop running the napari
plugin in Remote mode. For a single user the safe default is to bind to localhost and
reach it over an SSH tunnel, e.g.::

    # on the workstation
    voxtell-server --host 127.0.0.1 --port 1527

    # on the laptop
    ssh -N -L 1527:127.0.0.1:1527 workstation
    # then point the plugin at http://127.0.0.1:1527

Use ``--host 0.0.0.0`` (optionally with ``--api-key``) only on a trusted network.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from voxtell.inference.predictor import VoxTellPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VoxTell inference server (HTTP) for the napari remote client.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 for LAN).")
    parser.add_argument("--port", type=int, default=1527, help="Port to listen on.")
    parser.add_argument("-m", "--model", type=str, default=None,
                        help="VoxTell model directory. Defaults to $VOXTELL_MODEL, else the "
                             "published model is downloaded from Hugging Face.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Inference device (default: cuda).")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device id (default: 0).")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Local precomputed text-embedding bank (.npz).")
    parser.add_argument("--no-precomputed", action="store_true",
                        help="Do not download the published embedding bank.")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Require this bearer token on every request (default: none). "
                             "Falls back to the VOXTELL_API_KEY environment variable.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn

    from voxtell.server.app import make_app

    # Treat an empty/unset VOXTELL_MODEL as "no model given" (download the default).
    model = args.model or os.environ.get("VOXTELL_MODEL") or None
    if model is not None:
        model_path = Path(model).expanduser()
        if not (model_path / "plans.json").exists():
            raise FileNotFoundError(f"plans.json not found in model directory: {model_path}")
        model = str(model_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print(f"Loading VoxTell on {device} ...", file=sys.stderr)
    predictor = VoxTellPredictor(
        model_dir=model,
        device=device,
        embedding_bank=args.embeddings,
        use_precomputed_embeddings=not args.no_precomputed,
    )

    model_name = os.path.basename(model.rstrip("/")) if model else "voxtell (downloaded)"
    app = make_app(
        predictor,
        model_name=model_name,
        api_key=args.api_key or os.environ.get("VOXTELL_API_KEY"),
    )
    print(f"Serving on http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
