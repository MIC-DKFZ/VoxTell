"""
``voxtell-finetune`` -- fine-tune nnU-Net with VoxTell's pretrained encoder.

The CLI mirrors ``nnUNetv2_train``: positional ``dataset configuration fold``, plus
``-tr``/``-p``/``-pretrained_weights``/``-device``/``--c``/``--val``. See the nnU-Net
documentation for the meaning of those arguments. The only addition is that the chosen
trainer transfers VoxTell's encoder weights (from ``-pretrained_weights``, the
``VOXTELL_PRETRAINED`` env var, or the plans' ``pretrain_info`` block).

Set ``nnUNet_preprocessed`` and ``nnUNet_results`` as usual. Example:
    voxtell-finetune DATASET_ID 3d_fullres 0 -pretrained_weights /path/to/checkpoint_final.pth
"""
import argparse
import os

import torch


def run_finetuning(dataset_name_or_id, configuration, fold, trainer_name="VoxTellTrainer",
                   plans_identifier="nnUNetPlans", pretrained_weights=None,
                   device="cuda", continue_training=False, only_run_validation=False):
    from os.path import isfile, join
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    import voxtell.training as voxtell_training

    for var in ("nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(var) is None:
            raise EnvironmentError(f"{var} environment variable not set.")
    if continue_training and only_run_validation:
        raise RuntimeError("Cannot set --c and --val at the same time.")

    dataset_name = maybe_convert_to_dataset_name(
        int(dataset_name_or_id) if str(dataset_name_or_id).isdigit() else dataset_name_or_id)
    preprocessed = join(os.environ["nnUNet_preprocessed"], dataset_name)

    plans_file = join(preprocessed, f"{plans_identifier}.json")
    if not isfile(plans_file):
        raise FileNotFoundError(
            f"Plans file not found: {plans_file}\nRun nnUNetv2_plan_and_preprocess -d "
            f"{dataset_name_or_id} -pl {plans_identifier} first.")

    if pretrained_weights is not None:
        if not isfile(pretrained_weights):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_weights}")
        os.environ["VOXTELL_PRETRAINED"] = pretrained_weights

    plans = load_json(plans_file)
    plans["continue_training"] = continue_training
    trainer = getattr(voxtell_training, trainer_name)(
        plans=plans,
        configuration=configuration,
        fold=fold if fold == "all" else int(fold),
        dataset_json=load_json(join(preprocessed, "dataset.json")),
        device=torch.device(device),
    )

    # On resume / validation the weights come from the saved checkpoint, not the encoder transfer.
    if continue_training or only_run_validation:
        trainer.pretrained_checkpoint = None

    if continue_training:
        ckpt = next((join(trainer.output_folder, c) for c in
                     ("checkpoint_final.pth", "checkpoint_latest.pth")
                     if isfile(join(trainer.output_folder, c))), None)
        if ckpt is None:
            print("WARNING: --c set but no checkpoint found; starting a new training.")
        else:
            trainer.load_checkpoint(ckpt)
    elif only_run_validation:
        trainer.load_checkpoint(join(trainer.output_folder, "checkpoint_final.pth"))

    if only_run_validation:
        trainer.perform_actual_validation()
    else:
        trainer.run_training()


def main():
    p = argparse.ArgumentParser(description="Fine-tune a standard nnU-Net from VoxTell's pretrained "
                                            "encoder. Arguments mirror nnUNetv2_train.")
    p.add_argument("dataset_name_or_id", help="nnU-Net dataset name or integer ID")
    p.add_argument("configuration", help="nnU-Net configuration, e.g. 3d_fullres")
    p.add_argument("fold", help="Cross-validation fold (0-4 or 'all')")
    p.add_argument("-tr", default="VoxTellTrainer", help="Trainer class name (from voxtell.training)")
    p.add_argument("-p", default="nnUNetPlans", help="Plans identifier")
    p.add_argument("-pretrained_weights", default=None,
                   help="Path to the VoxTell checkpoint whose encoder weights are transferred "
                        "(overrides the plans' pretrain_info)")
    p.add_argument("-device", default="cuda", help="cuda / cpu / mps")
    p.add_argument("--c", action="store_true", help="Continue training from the latest checkpoint")
    p.add_argument("--val", action="store_true", help="Only run validation (requires checkpoint_final)")
    args = p.parse_args()

    run_finetuning(
        dataset_name_or_id=args.dataset_name_or_id, configuration=args.configuration, fold=args.fold,
        trainer_name=args.tr, plans_identifier=args.p, pretrained_weights=args.pretrained_weights,
        device=args.device, continue_training=args.c, only_run_validation=args.val,
    )


if __name__ == "__main__":
    main()
