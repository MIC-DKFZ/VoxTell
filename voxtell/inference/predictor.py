import os
import pydoc
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from nnunetv2.utilities.helpers import dummy_context, empty_cache

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.embedding_bank import download_embedding_bank, load_embedding_bank
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction


# Max number of prompts embedded by the text backbone in a single forward pass
_TEXT_EMBED_BATCH_SIZE = 32


class VoxTellPredictor:
    """
    Predictor for VoxTell segmentation model.
    
    This class handles loading the VoxTell model, preprocessing images,
    embedding text prompts, and performing sliding window inference to generate
    segmentation masks based on free-text anatomical descriptions.
    
    Attributes:
        device: PyTorch device for inference.
        network: The VoxTell model.
        tokenizer: Text tokenizer for prompt encoding.
        text_backbone: Text embedding model.
        patch_size: Patch size for sliding window inference.
        perform_everything_on_device: Keep all tensors on device during inference.
        max_text_length: Maximum text prompt length in tokens.
    """
    def __init__(self, model_dir: Optional[str] = None, device: torch.device = torch.device('cuda'),
                 text_encoding_model: str = 'Qwen/Qwen3-Embedding-4B',
                 embedding_bank: Optional[Union[str, Dict[str, np.ndarray]]] = None,
                 use_precomputed_embeddings: bool = True) -> None:
        """
        Initialize the VoxTell predictor.

        Args:
            model_dir: Path to model directory containing plans.json and
                fold_0/checkpoint_final.pth. If None (default), the path is read
                from the ``VOXTELL_MODEL`` environment variable.
            device: PyTorch device to use for inference (default: cuda).
            text_encoding_model: Pretrained text encoding model (Qwen/Qwen3-Embedding-4B).
            embedding_bank: Optional explicit bank of precomputed text embeddings,
                either a path to a ``.npz`` file or a ``{prompt: vector}`` dict.
                Prompts found in the bank skip the Qwen3 backbone. Overrides
                ``use_precomputed_embeddings``.
            use_precomputed_embeddings: If True (default) and no ``embedding_bank``
                is given, download the published bank from Hugging Face Hub in the
                background. If the download fails, a warning is printed and prompts 
                are embedded on the fly instead. Set False to skip the download entirely.

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
        """
        # Device setup
        self.device = device
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.normalization = ZScoreNormalization(intensityproperties={})

        # Predictor settings
        self.tile_step_size = 0.5
        self.perform_everything_on_device = True

        # Text embedding model loaded lazily
        self._text_encoding_model = text_encoding_model
        self.tokenizer = None
        self.text_backbone = None
        self.max_text_length = 8192

        # Precomputed embedding bank: a {prompt: float16 vector} dict, or None.
        # Embeddings computed on the fly are also cached here for the session.
        self.embedding_bank = self._resolve_embedding_bank(embedding_bank, use_precomputed_embeddings)

        # Resolve the model directory: explicit path, else the VOXTELL_MODEL env var.
        if model_dir is None:
            model_dir = os.environ.get('VOXTELL_MODEL')
        if model_dir is None:
            raise ValueError(
                "No model directory given. Pass model_dir=... or set the "
                "VOXTELL_MODEL environment variable to the model folder."
            )

        # Load network settings
        plans = load_json(join(model_dir, 'plans.json'))
        arch_kwargs = plans['configurations']['3d_fullres']['architecture']['arch_kwargs']
        self.patch_size = plans['configurations']['3d_fullres']['patch_size']

        arch_kwargs = dict(**arch_kwargs)
        for required_import_key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
            if arch_kwargs[required_import_key] is not None:
                arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])

        # Instantiate network
        network = VoxTellModel(
            input_channels=1,
            **arch_kwargs,
            decoder_layer=4,
            text_embedding_dim=2560,
            num_maskformer_stages=5,
            num_heads=32,
            query_dim=2048,
            project_to_decoder_hidden_dim=2048,
            deep_supervision=False
        )

        # Load weights
        checkpoint = torch.load(
            join(model_dir, 'fold_0', 'checkpoint_final.pth'),
            map_location=torch.device('cpu'),
            weights_only=False
        )

        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint['network_weights'])
        else:
            network._orig_mod.load_state_dict(checkpoint['network_weights'])
        
        network.eval()
        self.network = network

    def preprocess(self, data: np.ndarray) -> Tuple[torch.Tensor, Tuple, Tuple[int, ...]]:
        """
        Preprocess a single image for inference.
        
        This function preprocesses an image already in RAS orientation by performing
        cropping to non-zero regions and z-score normalization.
        
        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            
        Returns:
            Tuple containing:
                - Preprocessed image tensor
                - Bounding box of cropped region
                - Original image shape
        """

        if data.ndim == 3:
            data = data[None]  # add channel axis
        data = data.astype(np.float32)  # this creates a copy
        original_shape = data.shape[1:]
        data, _, bbox = crop_to_nonzero(data, None)
        data = self.normalization.run(data, None)
        data = torch.from_numpy(data)
        return data, bbox, original_shape
    
    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]) -> List[Tuple]:
        """
        Generate sliding window slicers for patch-based inference.
        
        Args:
            image_size: Shape of the input image.
            
        Returns:
            List of slice tuples for extracting patches.
        """
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1, (
                'if tile_size has less entries than image_size, '
                'len(tile_size) must be one shorter than len(image_size) '
                '(only dimension discrepancy of 1 allowed).'
            )
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size,
                                                     self.tile_step_size)
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size,
                                                     self.tile_step_size)
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.patch_size)]]))
        return slicers
    
    def _resolve_embedding_bank(
        self,
        embedding_bank: Optional[Union[str, Dict[str, np.ndarray]]],
        use_precomputed_embeddings: bool,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Resolve the bank argument into a ``{prompt: vector}`` dict or None.

        An explicit ``embedding_bank`` (path or dict) wins. Otherwise, if
        ``use_precomputed_embeddings`` is set, the published bank is downloaded;
        a failed download is handled gracefully
        """
        if embedding_bank is not None:
            if isinstance(embedding_bank, (str, Path)):
                return load_embedding_bank(str(embedding_bank))
            if isinstance(embedding_bank, dict):
                return dict(embedding_bank)
            raise TypeError(
                f"embedding_bank must be a path, dict, or None, got {type(embedding_bank)}"
            )
        if use_precomputed_embeddings:
            try:
                return download_embedding_bank()
            except Exception as exc:
                print(f"Note: could not download precomputed embeddings ({exc}); "
                      "prompts will be embedded with the text backbone.")
        return None

    def list_available_embeddings(self) -> List[str]:
        """Return the sorted prompt strings available as precomputed embeddings."""
        return sorted(self.embedding_bank) if self.embedding_bank is not None else []

    def _ensure_text_backbone(self) -> None:
        """Lazily load the tokenizer and text backbone on first use."""
        if self.text_backbone is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._text_encoding_model, padding_side='left'
            )
            self.text_backbone = AutoModel.from_pretrained(self._text_encoding_model).eval()

    @torch.inference_mode()
    def _compute_text_embeddings(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Embed text prompts with the Qwen3 backbone.

        Args:
            text_prompts: List of prompt strings to embed.

        Returns:
            Embeddings tensor of shape (num_prompts, embedding_dim) on ``self.device``.
        """
        self._ensure_text_backbone()
        self.text_backbone = self.text_backbone.to(self.device)

        # Embed all text prompts in chunks
        chunk_embeddings: List[torch.Tensor] = []
        for start in range(0, len(text_prompts), _TEXT_EMBED_BATCH_SIZE):
            chunk = text_prompts[start:start + _TEXT_EMBED_BATCH_SIZE]
            wrapped = wrap_with_instruction(chunk)
            text_tokens = self.tokenizer(
                wrapped,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
            text_embed = self.text_backbone(**text_tokens)
            chunk_embeddings.append(
                last_token_pool(
                    text_embed.last_hidden_state, text_tokens['attention_mask']
                ).float()
            )

        self.text_backbone = self.text_backbone.to('cpu')
        empty_cache(self.device)
        return torch.cat(chunk_embeddings, dim=0)

    @torch.inference_mode()
    def embed_text_prompts(self, text_prompts: Union[List[str], str]) -> torch.Tensor:
        """
        Embed text prompts into vector representations.

        Prompts present in the (optional) precomputed embedding bank are looked
        up directly; only the remaining prompts are passed through the Qwen3
        backbone. If every prompt is cached, the backbone is never loaded.

        Args:
            text_prompts: Single text prompt or list of text prompts.
        Returns:
            Text embeddings tensor of shape (1, num_prompts, embedding_dim).
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)

        embeddings: List[Optional[torch.Tensor]] = [None] * n_prompts
        to_compute: List[Tuple[int, str]] = []
        for i, prompt in enumerate(text_prompts):
            if self.embedding_bank is not None and prompt in self.embedding_bank:
                cached = np.asarray(self.embedding_bank[prompt], dtype=np.float32)
                embeddings[i] = torch.from_numpy(cached).to(self.device)
            else:
                to_compute.append((i, prompt))

        if to_compute:
            computed = self._compute_text_embeddings([prompt for _, prompt in to_compute])
            if self.embedding_bank is None:
                self.embedding_bank = {}
            for (idx, prompt), vec in zip(to_compute, computed):
                embeddings[idx] = vec
                # Cache for reuse within this session (e.g. repeated prompts).
                self.embedding_bank[prompt] = vec.detach().cpu().numpy().astype(np.float16)

        stacked = torch.stack(embeddings).to(self.device)
        return stacked.view(1, n_prompts, -1)

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform sliding window inference to generate segmentation logits.
        
        Args:
            input_image: Input image tensor of shape (C, X, Y, Z).
            text_embeddings: Text embeddings from embed_text_prompts.
            
        Returns:
            Predicted logits tensor.
            
        Raises:
            ValueError: If input_image is not 4D or not a torch.Tensor.
        """
        if not isinstance(input_image, torch.Tensor):
            raise ValueError(f"input_image must be a torch.Tensor, got {type(input_image)}")
        if input_image.ndim != 4:
            raise ValueError(
                f"input_image must be 4D (C, X, Y, Z), got shape {input_image.shape}"
            )
        
        self.network = self.network.to(self.device)

        empty_cache(self.device)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

            # if input_image is smaller than tile_size we need to pad it to tile_size.
            data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                       'constant', {'value': 0}, True, None)

            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

            predicted_logits = self._internal_predict_sliding_window_return_logits(
                data, text_embeddings, slicers, self.perform_everything_on_device
            )

            empty_cache(self.device)
            # Revert padding
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits
    
    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        text_embeddings: torch.Tensor,
        slicers: List[Tuple],
        do_on_device: bool = True,
    ) -> torch.Tensor:
        """
        Internal method for sliding window prediction with Gaussian weighting.
        
        Uses a producer-consumer pattern with threading to overlap data loading
        and model inference.
        
        Args:
            data: Preprocessed image data.
            text_embeddings: Text embeddings for prompts.
            slicers: List of slice tuples for patch extraction.
            do_on_device: If True, keep all tensors on GPU during computation.
            
        Returns:
            Aggregated prediction logits.
            
        Raises:
            RuntimeError: If inf values are encountered in predictions.
        """
        results_device = self.device if do_on_device else torch.device('cpu')

        def producer(data_tensor, slicer_list, queue):
            """Producer thread that loads patches into queue."""
            for slicer in slicer_list:
                patch = torch.clone(
                    data_tensor[slicer][None],
                    memory_format=torch.contiguous_format
                ).to(self.device)
                queue.put((patch, slicer))
            queue.put('end')

        empty_cache(self.device)

        # move data to device
        data = data.to(results_device)
        queue = Queue(maxsize=2)
        t = Thread(target=producer, args=(data, slicers, queue))
        t.start()

        # preallocate arrays
        predicted_logits = torch.zeros((text_embeddings.shape[1], *data.shape[1:]),
                                        dtype=torch.half,
                                        device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        gaussian = compute_gaussian(
            tuple(self.patch_size),
            sigma_scale=1. / 8,
            value_scaling_factor=10,
            device=results_device
        )

        with tqdm(desc=None, total=len(slicers)) as pbar:
            while True:
                item = queue.get()
                if item == 'end':
                    queue.task_done()
                    break
                patch, tile_slice = item
                prediction = self.network(patch, text_embeddings)[0].to(results_device)
                prediction *= gaussian
                predicted_logits[tile_slice] += prediction
                n_predictions[tile_slice[1:]] += gaussian
                queue.task_done()
                pbar.update()
        queue.join()

        # Normalize by number of predictions per voxel
        torch.div(predicted_logits, n_predictions, out=predicted_logits)
        
        # Check for inf values
        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError(
                'Encountered inf in predicted array. Aborting... '
                'If this problem persists, reduce value_scaling_factor in '
                'compute_gaussian or increase the dtype of predicted_logits to fp32.'
            )
        return predicted_logits

    def predict_single_image(
        self,
        data: np.ndarray,
        text_prompts: Union[str, List[str], None] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Predict segmentation masks for a single image with text prompts.

        This is the main prediction method that orchestrates preprocessing,
        text embedding, sliding window inference, and postprocessing.

        Args:
            data: Image data in RAS orientation (3D or 4D with channel dimension).
            text_prompts: Single text prompt or list of text prompts describing
                anatomical structures to segment. Ignored if ``text_embeddings``
                is provided.
            text_embeddings: Optional precomputed embeddings of shape
                (1, num_prompts, embedding_dim), as returned by
                :meth:`embed_text_prompts`. Pass this to reuse one embedding pass
                across many images (see :meth:`predict_from_files`).

        Returns:
            Segmentation masks as numpy array of shape (num_prompts, X, Y, Z)
            with binary values (0 or 1) indicating the segmented regions.
        """
        if text_embeddings is None:
            if text_prompts is None:
                raise ValueError("Provide either text_prompts or text_embeddings.")
            text_embeddings = self.embed_text_prompts(text_prompts)

        # Preprocess image
        data, bbox, orig_shape = self.preprocess(data)

        # Predict segmentation logits
        prediction = self.predict_sliding_window_return_logits(data, text_embeddings).to('cpu')

        # Postprocess logits to get binary segmentation masks
        with torch.no_grad():
            prediction = torch.sigmoid(prediction.float()) > 0.5
        
        segmentation_reverted_cropping = np.zeros(
            [prediction.shape[0], *orig_shape],
            dtype=np.uint8
        )
        segmentation_reverted_cropping = insert_crop_into_image(
            segmentation_reverted_cropping, prediction, bbox
        )

        return segmentation_reverted_cropping

    _NIFTI_SUFFIXES = ('.nii.gz', '.nii')

    @staticmethod
    def _safe_prompt_name(prompt: str) -> str:
        """Turn a prompt into a filesystem-safe filename fragment."""
        safe = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in prompt)
        return safe.replace(' ', '_')

    @classmethod
    def _case_name(cls, file_path: Union[str, Path]) -> str:
        """Strip the NIfTI extension from a filename to get the case identifier."""
        name = os.path.basename(str(file_path))
        for suffix in cls._NIFTI_SUFFIXES:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return os.path.splitext(name)[0]

    @classmethod
    def _is_nifti(cls, name: str) -> bool:
        return any(name.endswith(suffix) for suffix in cls._NIFTI_SUFFIXES)

    @classmethod
    def _resolve_input_files(
        cls, inputs: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[str]:
        """Resolve inputs into an ordered list of NIfTI paths.

        ``inputs`` is **either** a single folder (expanded to all its NIfTI
        files, sorted by name) **or** one or more file paths (kept in the given
        order, de-duplicated). Folders and files must not be mixed — pick one.
        Paths may be absolute or relative to the current working directory.
        """
        if isinstance(inputs, (str, Path)):
            inputs = [inputs]
        entries = [Path(e) for e in inputs]

        if any(e.is_dir() for e in entries):
            if len(entries) != 1:
                raise ValueError(
                    "Provide either a single folder or a list of files, not a mix "
                    "of folders and other paths."
                )
            folder = entries[0]
            files = sorted(
                str(p) for p in folder.iterdir() if p.is_file() and cls._is_nifti(p.name)
            )
            if not files:
                raise FileNotFoundError(f"No NIfTI files found in folder: {folder}")
            return files

        seen = set()
        unique = [str(e) for e in entries if not (str(e) in seen or seen.add(str(e)))]
        if not unique:
            raise FileNotFoundError("No input images given.")
        return unique

    def predict_from_files(
        self,
        inputs: Union[str, Path, Sequence[Union[str, Path]]],
        output_folder: Union[str, Path],
        text_prompts: Union[str, List[str]],
        save_combined: bool = False,
        overwrite: bool = True,
        verbose: bool = False,
    ) -> List[str]:
        """
        Segment many images with the **same** prompts, embedding them only once.

        ``inputs`` is a single folder or one/more file paths (see
        :meth:`_resolve_input_files`). For per-image prompts use
        :meth:`predict_from_jobs`.

        Args:
            inputs: A single folder, or one or more NIfTI file paths.
            output_folder: Directory to write segmentation files into.
            text_prompts: Prompt or list of prompts applied to every image.
            save_combined: If True, write one multi-label file per image
                (label ``i+1`` = prompt ``i``; later prompts overwrite overlaps).
                Otherwise write one binary file per prompt.
            overwrite: If False, skip images whose expected outputs already exist.
            verbose: Print per-image progress.

        Returns:
            List of written output file paths.
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        files = self._resolve_input_files(inputs)
        jobs = [{"image": f, "prompts": text_prompts} for f in files]
        return self.predict_from_jobs(
            jobs, output_folder, save_combined=save_combined,
            overwrite=overwrite, verbose=verbose,
        )

    def predict_from_jobs(
        self,
        jobs: Sequence[Dict[str, Union[str, List[str]]]],
        output_folder: Union[str, Path],
        save_combined: bool = False,
        overwrite: bool = True,
        verbose: bool = False,
    ) -> List[str]:
        """
        Segment images where **each image has its own prompts**.

        The union of all prompts across jobs is embedded once and reused.

        Args:
            jobs: Sequence of dicts ``{"image": <path>, "prompts": [str, ...]}``.
                ``prompts`` may also be a single string.
            output_folder: Directory to write segmentation files into.
            save_combined: One multi-label file per image instead of one per prompt.
            overwrite: If False, skip images whose expected outputs already exist.
            verbose: Print per-image progress.

        Returns:
            List of written output file paths.
        """
        # Normalize jobs to (image, prompts-list).
        norm: List[Tuple[str, List[str]]] = []
        for job in jobs:
            prompts = job["prompts"]
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
            if not prompts:
                raise ValueError(f"Job for {job.get('image')!r} has no prompts.")
            norm.append((str(job["image"]), prompts))
        if not norm:
            raise ValueError("No jobs to predict.")

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        reader_writer = NibabelIOWithReorient()

        # Images from different folders can share a basename and would then write
        # to the same output file, throwing a warning
        case_names = [self._case_name(img) for img, _ in norm]
        duplicates = sorted({c for c in case_names if case_names.count(c) > 1})
        if duplicates:
            print(
                f"WARNING: {len(duplicates)} case name(s) occur multiple times across the "
                f"inputs ({duplicates[:5]}{' ...' if len(duplicates) > 5 else ''}). "
                "Their outputs will overwrite each other in the output folder."
            )

        # Embed the union of all prompts exactly once, then assemble each job's
        # embeddings by lookup (independent of the cache setting).
        unique_prompts: List[str] = []
        seen = set()
        for _, prompts in norm:
            for p in prompts:
                if p not in seen:
                    seen.add(p)
                    unique_prompts.append(p)
        union = self.embed_text_prompts(unique_prompts)  # (1, U, dim)
        prompt_to_vec = {p: union[0, i] for i, p in enumerate(unique_prompts)}

        todo: List[Tuple[str, List[str]]] = []
        for image, prompts in norm:
            case = self._case_name(image)
            expected = self._expected_outputs(output_folder, case, prompts, save_combined)
            if not overwrite and all(os.path.isfile(p) for p in expected):
                if verbose:
                    print(f"Skipping {case} (outputs exist)")
                continue
            todo.append((image, prompts))

        written: List[str] = []
        for image, prompts in tqdm(todo, desc='Images',
                                   disable=verbose or len(todo) <= 1):
            case = self._case_name(image)
            if verbose:
                print(f"Predicting {case} with prompts {prompts} ...")
            img, props = reader_writer.read_images([image])
            job_embeddings = torch.stack([prompt_to_vec[p] for p in prompts]).view(
                1, len(prompts), -1
            )
            segmentation = self.predict_single_image(
                img, text_embeddings=job_embeddings
            )
            written.extend(
                self._save_segmentation(
                    segmentation, output_folder, case, prompts, props,
                    save_combined, reader_writer,
                )
            )
        return written

    def _expected_outputs(
        self, output_folder: Path, case: str, text_prompts: List[str], save_combined: bool
    ) -> List[str]:
        """Output file paths that prediction of ``case`` will produce."""
        if save_combined:
            return [str(output_folder / f"{case}.nii.gz")]
        return [
            str(output_folder / f"{case}_{self._safe_prompt_name(p)}.nii.gz")
            for p in text_prompts
        ]

    def _save_segmentation(
        self, segmentation: np.ndarray, output_folder: Path, case: str,
        text_prompts: List[str], props: dict, save_combined: bool,
        reader_writer: NibabelIOWithReorient,
    ) -> List[str]:
        """Write a prediction to disk and return the written paths."""
        written: List[str] = []
        if save_combined:
            combined = np.zeros_like(segmentation[0], dtype=np.uint8)
            for i, seg in enumerate(segmentation):
                combined[seg > 0] = i + 1
            out_path = str(output_folder / f"{case}.nii.gz")
            reader_writer.write_seg(combined, out_path, props)
            written.append(out_path)
        else:
            for i, prompt in enumerate(text_prompts):
                out_path = str(output_folder / f"{case}_{self._safe_prompt_name(prompt)}.nii.gz")
                reader_writer.write_seg(segmentation[i], out_path, props)
                written.append(out_path)
        return written


if __name__ == '__main__':
    from pathlib import Path
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    # Default paths - modify these as needed
    DEFAULT_IMAGE_PATH = "/path/to/your/image.nii.gz"
    DEFAULT_MODEL_DIR = "/path/to/your/model/directory"
    
    # Configuration
    image_path = DEFAULT_IMAGE_PATH
    model_dir = DEFAULT_MODEL_DIR
    text_prompts = ["liver", "right kidney", "left kidney", "spleen"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load image
    img, props = NibabelIOWithReorient().read_images([image_path])
    
    # Initialize predictor and run inference
    predictor = VoxTellPredictor(model_dir=model_dir, device=device)
    voxtell_seg = predictor.predict_single_image(img, text_prompts)
    
    # Visualize results, we reccommend using napari for 3D visualization
    import napari
    viewer = napari.Viewer()
    viewer.add_image(img, name='image')
    for i, prompt in enumerate(text_prompts):
        viewer.add_labels(voxtell_seg[i], name=f'voxtell_{prompt}')
    napari.run()