# AGENTS.md

VoxTell — free-text promptable 3D medical image segmentation (CVPR 2026, [arXiv:2511.11450](https://arxiv.org/abs/2511.11450)).
A frozen Qwen3-Embedding-4B encodes the prompt; a ResEnc-L 3D U-Net encodes the volume; a DETR-style
prompt decoder turns the prompt into mask embeddings that are fused into the image decoder at five
scales. Inference is nnU-Net-style sliding window. Weights live on HF (`mrokuss/VoxTell`), not here.

## Commands

```bash
# Setup (Python >=3.10; torch 2.9.x is excluded — install torch first, matched to your CUDA)
conda create -n voxtell python=3.12 && conda activate voxtell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,server]"

# Smoke checks (no GPU and no local model needed; --list-embeddings hits HF)
python -c "import voxtell; print(voxtell.__version__)"
voxtell-predict --help
voxtell-predict --list-embeddings | head

# Run
voxtell-predict -i case.nii.gz -o out -p "liver" "spleen"     # or -i folder/, or --jobs jobs.json
voxtell-server --host 127.0.0.1 --port 1527                   # needs [server] extra
voxtell-finetune DATASET_ID 3d_fullres 0 -pretrained_weights <model>/fold_0/checkpoint_final.pth
```

**There is no test suite, no CI, no typecheck, and no lint/format config.** `black`/`ruff` are listed in
`[dev]` but nothing is configured and the code is *not* black-formatted (quote style differs per module
by history). Do not run a repo-wide reformat — it buries the real diff. Verify changes by running the
CLI end-to-end on a real NIfTI, and the server by hitting `/healthz` and one `/jobs` round-trip.

## Layout

```
voxtell/
  inference/predictor.py            VoxTellPredictor — the core API. Model loading, preprocessing,
                                    text embedding, sliding window, predict_single_image /
                                    predict_from_files / predict_from_jobs. Biggest file; start here.
  inference/predict_from_raw_data.py  `voxtell-predict` argparse CLI (thin wrapper over the predictor)
  model/voxtell_model.py            VoxTellModel (encoder + prompt decoder) and VoxTellDecoder (fusion)
  model/transformer.py              DETR-derived transformer decoder used as the prompt decoder
  training/voxtell_trainer.py       VoxTellTrainer(_noMirroring) — nnU-Net trainers, encoder transfer
  training/run_finetuning.py        `voxtell-finetune` CLI (mirrors nnUNetv2_train)
  server/app.py                     FastAPI routes: /images, /jobs, /jobs/{id}/events|result|cancel
  server/runner.py                  GPU-side engine: prompt batching, OOM fallbacks, keep-largest-CC
  server/serialization.py           blosc2 array wire format (duplicated in the napari client)
  server/__main__.py                `voxtell-server` CLI
  utils/text_embedding.py           instruction wrapping + last-token pooling for Qwen3
  utils/embedding_bank.py           precomputed {prompt: fp16 vector} .npz bank, local or from HF
readme.md                           lowercase on purpose; it is the packaged long description
```

## Conventions

**Orientation is the #1 correctness trap.** Images must reach the predictor in RAS, read with
nnU-Net's `NibabelIOWithReorient` — the exact reader used in training. Never swap in plain
`nibabel`/`SimpleITK` for reading or writing, and always carry the returned `props` through to
`write_seg`. A wrong orientation fails silently: "liver" segments the spleen, left/right flip.
The server reorients once on upload and is the single source of truth for its clients.

**Prompts are lowercased** before embedding (`embed_text_prompts`), and bank keys are lowercase to
match. Route every new prompt path through `embed_text_prompts` rather than embedding ad hoc.

**Model resolution order is explicit arg → `$VOXTELL_MODEL` → HF download**, and it is repeated
identically in the predictor, `voxtell-predict` and `voxtell-server`. Any new entry point must keep
that order. Same for the embedding bank: explicit path/dict → published HF bank → backbone.

**Embed once, reuse across images.** `predict_from_jobs` embeds the *union* of all prompts (of the
jobs not skipped by `overwrite=False`) and hands per-job slices to
`predict_single_image(text_embeddings=...)`. New batch paths must not re-embed per image; the 4B
backbone dominates runtime for short jobs. It also prefetches the next image and writes finished masks
in background threads (at most one read and one write in flight), since gzip I/O of large volumes
takes seconds per image.

**The text backbone is lazy, bf16 on CUDA, and moved back to CPU after every embed call.** That is
what keeps VoxTell usable on ~8 GB GPUs. Do not make it resident or float32 on CUDA.

**Architecture args must match the checkpoint.** Only `arch_kwargs` and `patch_size` come from
`plans.json`; the rest (`decoder_layer=4`, `text_embedding_dim=2560`, `num_maskformer_stages=5`,
`num_heads=32`, `query_dim=2048`, `project_to_decoder_hidden_dim=2048`) are hardcoded in
`VoxTellPredictor.__init__` and are part of the released weights. Changing them requires a new
checkpoint.

**Patch size is effectively locked to 192³.** `VoxTellModel.DECODER_CONFIGS` hardcodes the spatial
shape of every stage, and `pos_embed` is a fixed-size registered buffer built from the
`decoder_layer` entry. A `plans.json` with a different patch size will not work — this is why
`VoxTellTrainer` forces 192³ when the dataset's plans are not divisible by 32.

**`seg_layers` are built even when `deep_supervision=False`** so deep-supervision-trained checkpoints
still load with `strict=True`. It looks like dead code. It is not — leave it.

**No resampling, by design.** Images are cropped to nonzero and z-score normalized, nothing else.
Don't add a resampling step; document unusual-spacing caveats in the readme instead.

**Heavy and optional dependencies are imported lazily.** `fastapi`/`uvicorn`/`blosc2` live behind
the `[server]` extra and are reachable only from `voxtell.server`; `scipy` and `napari` are never
declared at all; `huggingface_hub` is declared but imported inside the download functions.
`import voxtell.inference.predictor` must keep working on the base install alone.

**`server/serialization.py` is duplicated verbatim in the napari-voxtell client.** Changing the framing
means bumping `_MAGIC` and updating both copies; the modules are kept dependency-light (numpy +
blosc2 only) precisely so the laptop client stays torch-free.

**Server jobs are serialized on purpose.** `RemoteInferenceEngine._lock` guards all GPU work because
the predictor is shared and mutable (`perform_everything_on_device` is flipped during OOM fallback).
Don't run jobs concurrently. Likewise, every exit from the sliding-window consumer loop (done, cancel
via `progress_callback` returning `False`, or an error such as an OOM in the forward pass) must stop
and drain the patch producer in the `finally` — otherwise the producer thread blocks forever holding
GPU tensors and interpreter exit hangs (see the comments there).

**Fine-tuning transfers `encoder.*` only.** `VoxTellTrainer.build_network_architecture` hardcodes the
6-stage ResEnc-L encoder independently of the dataset's plans so the pretrained weights load with
`strict=True`; keep it in sync with the checkpoint. The trainer also prints the VoxTell citation
alongside nnU-Net's — keep that.

**Docs and artifacts.** Update the CLI flag table in `readme.md` whenever argparse changes; it is the
only reference for the flags. Never commit weights, `.npz` banks, or NIfTI data — `.gitignore` blocks
them and model artifacts belong on Hugging Face. Match the existing docstring style: module
docstrings say *why*, function docstrings use Google-style `Args:`/`Returns:`/`Raises:`.

**Releases follow nnU-Net (one `vX.Y.Z` tag per PyPI release), and `main` is always the latest
release.** Code changes reach `main` only as part of a release, so a GitHub install never claims a
version whose code differs from PyPI. Docs-only changes may land at any time. A release:
1. Bumps `version` in `pyproject.toml`.
2. Builds from that exact commit: `rm -rf dist && uv build && uvx twine check dist/*`.
3. Publishes to PyPI with `uv publish --check-url https://pypi.org/simple/`. If an upload fails
   midway, rerun it without rebuilding, because PyPI rejects a filename it has seen with different
   bytes.
4. Only then pushes `main` and a `vX.Y.Z` tag on the built commit. The readme tells users to
   `pip install` the new version, so it must not reach GitHub first.

napari-voxtell is versioned independently (like napari-nninteractive and nnInteractive) and pins
`voxtell>=` the oldest release it needs. When both change, release voxtell first.
