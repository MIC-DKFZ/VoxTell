"""
Precomputed text-embedding bank for VoxTell.

VoxTell embeds free-text prompts with a frozen Qwen3-Embedding-4B backbone before 
running the image decoder. Embedding is deterministic for a given prompt string 
(same instruction wrapping + last-token pooling), so prompts that are known ahead 
of time can be embedded once and reused instead of paying the 4B backbone cost on
every call.

A "bank" is simply a ``{prompt: embedding}`` mapping. It is stored on disk as a
single ``.npz`` with two arrays:
- ``labels``: unicode string array of length ``N`` (the prompt strings)
- ``embeddings``: ``float16`` array of shape ``(N, EMBEDDING_DIM)``
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

EMBEDDING_DIM = 2560

DEFAULT_HF_REPO = "mrokuss/VoxTell"
DEFAULT_EMBEDDING_MODEL = "voxtell_v1.1"


def hf_embedding_path(model_name: str = DEFAULT_EMBEDDING_MODEL) -> str:
    """Return the Hugging Face repo path of the bank for a given model version."""
    return f"embeddings/{model_name}/text_embeddings.npz"


def load_embedding_bank(path: str) -> Dict[str, np.ndarray]:
    """Load a ``{prompt: float16 vector}`` bank from a ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        labels = data["labels"]
        embeddings = data["embeddings"]
    return {str(label): embeddings[i] for i, label in enumerate(labels)}


def save_embedding_bank(bank: Dict[str, np.ndarray], path: str) -> str:
    """Save a ``{prompt: vector}`` bank to ``path`` as a compressed ``.npz``.

    Vectors are stored as ``float16``. Used by the bank-builder script; the
    predictor never writes banks.
    """
    labels = list(bank.keys())
    if labels:
        embeddings = np.stack([np.asarray(bank[l], dtype=np.float16).reshape(-1) for l in labels])
    else:
        embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float16)
    np.savez_compressed(path, labels=np.array(labels, dtype=np.str_), embeddings=embeddings)
    return path


def download_embedding_bank(
    repo_id: str = DEFAULT_HF_REPO,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    filename: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Download and load the published embedding bank from Hugging Face Hub.

    Fetches ``embeddings/<model_name>/text_embeddings.npz`` from ``repo_id`` by
    default; pass ``filename`` to override the path.
    """
    from huggingface_hub import hf_hub_download

    if filename is None:
        filename = hf_embedding_path(model_name)
    path = hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, cache_dir=cache_dir
    )
    return load_embedding_bank(path)
