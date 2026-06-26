#!/usr/bin/env python3
"""
Command-line entrypoint for VoxTell segmentation prediction.

This script provides a CLI interface to run VoxTell predictions on medical images
with free-text prompts. It accepts a single image, a list of images, or a whole
folder, and embeds the text prompts only once across all inputs. Prompts can be
the same for every image (-p) or specified per image (--jobs).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from voxtell.inference.predictor import VoxTellPredictor
from voxtell.utils.embedding_bank import download_embedding_bank, load_embedding_bank


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Inputs (-i): EITHER a single folder (all NIfTI files in it) OR one or more files.
Prompts: choose ONE of
  -p / --prompts        same prompts for every image (simplest)
  --jobs                a JSON [{"image":..,"prompts":[..]}, ..] binding each
                        image to its own prompts (image paths come from the file)

The model directory comes from -m/--model or the VOXTELL_MODEL environment
variable; if neither is set, the default model (voxtell_v1.1) is downloaded from
Hugging Face.

Precomputed embeddings are downloaded automatically and used behind the scenes;
prompts not in the bank are embedded with the text backbone. Use --no-precomputed
to disable the download.

Examples:
  # One-time setup: point VOXTELL_MODEL at your model directory
  export VOXTELL_MODEL=/path/to/model

  # Single image, single prompt (saves output_folder/case001_liver.nii.gz)
  voxtell-predict -i case001.nii.gz -o out -p "liver"

  # Whole folder, same prompts (text embedded once, reused for every image)
  voxtell-predict -i images_folder -o out -p "liver" "spleen"

  # An explicit list of files, same prompts for all
  voxtell-predict -i a.nii.gz b.nii.gz c.nii.gz -o out -p "liver"

  # Per-image prompts via a jobs file (images + prompts together, -i not used)
  voxtell-predict -o out --jobs jobs.json

  # Point at a model explicitly (overrides VOXTELL_MODEL for this run)
  voxtell-predict -i case001.nii.gz -o out -m /path/to/model -p "liver"

  # Use a local embedding bank, or turn the automatic download off
  voxtell-predict -i case001.nii.gz -o out -p "liver" --embeddings bank.npz
  voxtell-predict -i case001.nii.gz -o out -p "liver" --no-precomputed

  # List which prompts are available as precomputed embeddings, then exit
  voxtell-predict --list-embeddings
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        nargs='+',
        help='Input images: EITHER a single folder (all NIfTI files in it) OR one '
             'or more NIfTI file paths (.nii/.nii.gz, absolute or relative to the '
             'current directory). Do not mix a folder with individual files. '
             'Not used with --jobs.'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Path to output folder where segmentation files will be saved'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        help='Path to VoxTell model directory (plans.json and fold_0/). If '
             'omitted, the VOXTELL_MODEL environment variable is used, or the '
             'default model (voxtell_v1.1) is downloaded from Hugging Face.'
    )

    parser.add_argument(
        '-p', '--prompts',
        type=str,
        nargs='+',
        help='Text prompt(s) applied to EVERY image (e.g., "liver" "spleen"). '
             'Simplest option for same-prompts-for-all.'
    )

    parser.add_argument(
        '--jobs',
        type=str,
        default=None,
        help='Per-image prompts as a JSON string or .json file: a list of '
             '{"image": <path>, "prompts": [..]} objects. Use this when each '
             'image needs different prompts. Images come from the file, so -i '
             'is not used.'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0)'
    )
    
    parser.add_argument(
        '--save-combined',
        action='store_true',
        help='Save all prompts in a single multi-label file per image (WARNING: '
             'overlapping structures will be overwritten by later prompts)'
    )

    parser.add_argument(
        '--embeddings',
        type=str,
        default=None,
        help='Path to a local precomputed text-embedding bank (.npz). Prompts '
             'found in the bank skip the text backbone. Overrides the automatic '
             'download.'
    )

    parser.add_argument(
        '--no-precomputed',
        action='store_true',
        help='Do not download the published precomputed embedding bank; embed '
             'every prompt with the text backbone instead.'
    )

    parser.add_argument(
        '--list-embeddings',
        action='store_true',
        help='List the prompts available in the embedding bank (local --embeddings '
             'if given, otherwise the published bank) and exit.'
    )

    parser.add_argument(
        '--no-overwrite',
        action='store_true',
        help='Skip images whose output files already exist.'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def _load_json_arg(value: str):
    """Parse a JSON CLI value that is either a path to a .json file or an inline JSON string."""
    if os.path.isfile(value):
        with open(value) as handle:
            return json.load(handle)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not read JSON (not an existing file and not valid JSON): {value!r}"
        ) from exc


def _run() -> int:
    """Main entrypoint body (wrapped by :func:`main` for broken-pipe handling)."""
    args = parse_args()

    # --list-embeddings: just inspect the bank and exit (no model needed).
    if args.list_embeddings:
        try:
            bank = (load_embedding_bank(args.embeddings) if args.embeddings
                    else download_embedding_bank())
        except Exception as exc:
            print(f"Error: could not load embedding bank ({exc})", file=sys.stderr)
            return 1
        # Single write; BrokenPipeError (e.g. piping into `head`) handled in __main__.
        print(f"{len(bank)} precomputed embeddings available:\n" +
              "\n".join(f"  {label}" for label in sorted(bank)))
        return 0

    # Exactly one prompt source must be given.
    prompt_sources = [
        name for name, val in
        (('-p/--prompts', args.prompts), ('--jobs', args.jobs)) if val
    ]
    if len(prompt_sources) != 1:
        print("Error: provide exactly one of -p/--prompts or --jobs "
              f"(got: {prompt_sources or 'none'})", file=sys.stderr)
        return 1

    # Output is always required; -i is required unless --jobs is used. The model
    # comes from -m/--model or, if omitted, the VOXTELL_MODEL env variable.
    missing = [name for name, val in (('--output', args.output),) if not val]
    if not args.jobs and not args.input:
        missing.append('--input')
    if missing:
        print(f"Error: the following arguments are required: {', '.join(missing)}",
              file=sys.stderr)
        return 1

    # Model from -m/--model or VOXTELL_MODEL; if neither is set, model stays None
    # and VoxTellPredictor downloads the default model from Hugging Face (cached).
    model = args.model or os.environ.get('VOXTELL_MODEL')

    if args.jobs and args.input:
        print("Warning: -i/--input is ignored when --jobs is given", file=sys.stderr)

    # Validate that -i paths exist
    for entry in (args.input or []):
        if not Path(entry).exists():
            raise FileNotFoundError(f"Input path does not exist: {entry}")

    # When a model is given it must be a local directory; validate it. If omitted,
    # model stays None and VoxTellPredictor downloads the default model.
    if model is not None:
        model_path = Path(model).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {model_path}")
        if not (model_path / 'plans.json').exists():
            raise FileNotFoundError(f"plans.json not found in model directory: {model_path}")
        if not (model_path / 'fold_0' / 'checkpoint_final.pth').exists():
            raise FileNotFoundError(f"checkpoint_final.pth not found in {model_path / 'fold_0'}")
        model = str(model_path)
    
    # Setup device
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{args.gpu}')
            if args.verbose:
                print(f"Using GPU: {args.gpu} ({torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device('cpu')
        if args.verbose:
            print("Using CPU")
    if args.verbose:
        print(f"Loading VoxTell model from: {model}" if model else
              "No model given; downloading the default VoxTell model from Hugging Face")

    predictor = VoxTellPredictor(
        model_dir=model,
        device=device,
        embedding_bank=args.embeddings,
        use_precomputed_embeddings=not args.no_precomputed,
    )

    if args.save_combined:
        print("\nNOTE: --save-combined writes one multi-label file per image; "
              "overlapping structures are overwritten by later prompts.\n")

    common = dict(
        output_folder=args.output,
        save_combined=args.save_combined,
        overwrite=not args.no_overwrite,
        verbose=args.verbose,
    )

    if args.jobs:
        # Per-image prompts from a jobs JSON: [{"image":.., "prompts":[..]}, ..]
        jobs = _load_json_arg(args.jobs)
        if not isinstance(jobs, list) or not all(isinstance(j, dict) for j in jobs):
            print("Error: --jobs must be a JSON list of {\"image\":.., \"prompts\":[..]} objects",
                  file=sys.stderr)
            return 1
        for job in jobs:
            if 'image' not in job or 'prompts' not in job:
                print("Error: each --jobs entry needs 'image' and 'prompts'", file=sys.stderr)
                return 1
            if not Path(job['image']).exists():
                raise FileNotFoundError(f"Job image does not exist: {job['image']}")
        written = predictor.predict_from_jobs(jobs, **common)
    else:
        # -p/--prompts: same prompts for all images.
        written = predictor.predict_from_files(inputs=args.input, text_prompts=args.prompts, **common)

    print(f"\nPrediction completed successfully! Wrote {len(written)} file(s).")
    return 0


def main() -> int:
    """CLI entrypoint with graceful handling of pipes that close early."""
    try:
        code = _run()
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        import os
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == '__main__':
    sys.exit(main())
