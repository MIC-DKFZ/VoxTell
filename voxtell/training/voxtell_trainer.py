"""
VoxTell encoder-transfer fine-tuning for nnU-Net.

`VoxTellTrainer` transfers only VoxTell's pretrained image encoder (the `encoder.*`
weights) into a standard nnU-Net and fine-tunes for multi-class segmentation; the decoder
trains from scratch. Configuration:

  * 2-stage schedule:  warmup_all (linear LR warmup) -> train (PolyLR with offset)
  * deep supervision OFF
  * batch size 2

The encoder architecture is built by the trainer (not read from the plans), so any
preprocessed dataset works without editing the plans. The pretrained checkpoint is taken
from the ``VOXTELL_PRETRAINED`` env var / the ``-pretrained_weights`` CLI flag, or the plans'
``pretrain_info`` block.
"""
import os

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnunetv2.training.lr_scheduler.warmup import Lin_incr_LRScheduler, PolyLRScheduler_offset
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache


class VoxTellTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        cfg = plans["configurations"][configuration]
        cfg["batch_size"] = 2
        # VoxTell's encoder downsamples 32x, so the patch must be divisible by 32.
        # Keep the plans' patch when it already is; otherwise use VoxTell's 192^3.
        if any(p % 32 for p in cfg["patch_size"]):
            cfg["patch_size"] = [192, 192, 192]
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False
        self.warmup_duration_whole_net = 50
        self.training_stage = None
        adaptation_info = self.plans_manager.plans.get("pretrain_info", {}) or {}
        self.pretrained_checkpoint = (os.environ.get("VOXTELL_PRETRAINED")
                                      or adaptation_info.get("checkpoint_path"))
        # nnU-Net already prints its own citation in super().__init__(); add VoxTell's.
        self.print_to_log_file(
            "\n#######################################################################\n"
            "Please also cite VoxTell when using the pretrained encoder:\n"
            "Rokuss, M., Langenberg, M., Kirchhoff, Y., Isensee, F., Hamm, B., Ulrich, C., "
            "Regnery, S., Bauer, L., Katsigiannopulos, E., Norajitra, T., & Maier-Hein, K. (2026). "
            "VoxTell: Free-text promptable universal 3D medical image segmentation. CVPR 2026, 37538-37557.\n"
            "#######################################################################\n",
            also_print_to_console=True, add_timestamp=False)

    # --------------------------------- network ---------------------------------- #
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        # Always VoxTell's encoder (6-stage ResEnc-L), independent of the dataset's plans,
        # so the pretrained encoder weights load. Only in/out channels are dataset-specific.
        n_stages = 6
        return ResidualEncoderUNet(
            input_channels=num_input_channels, n_stages=n_stages,
            features_per_stage=[32, 64, 128, 256, 320, 320], conv_op=torch.nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * n_stages,
            strides=[[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
            n_blocks_per_stage=[1, 3, 4, 6, 6, 6], num_classes=num_output_channels,
            n_conv_per_stage_decoder=[1, 1, 1, 1, 1], conv_bias=True,
            norm_op=torch.nn.InstanceNorm3d, norm_op_kwargs={"eps": 1e-5, "affine": True},
            nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision,
        )

    # ------------------------------- weight loading ----------------------------- #
    def _unwrap(self):
        net = self.network.module if isinstance(self.network, DDP) else self.network
        return net._orig_mod if isinstance(net, OptimizedModule) else net

    def initialize(self):
        super().initialize()
        if self.pretrained_checkpoint:
            self._load_encoder_weights(self.pretrained_checkpoint)
        else:
            self.print_to_log_file("No pretrained checkpoint -- training encoder from scratch.")

    def _load_encoder_weights(self, path: str):
        """Copy only the pretrained ``encoder.*`` weights from a VoxTell checkpoint."""
        ckp = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckp["network_weights"]
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}

        # Adapt the stem conv from 1 pretrained input channel to N (repeat + average);
        # the stem conv weight is stored under two aliased keys, so adapt both.
        n_ch = self.num_input_channels
        for stem_key in ("stem.convs.0.conv.weight", "stem.convs.0.all_modules.0.weight"):
            if stem_key in enc and n_ch > 1 and enc[stem_key].shape[1] == 1:
                enc[stem_key] = enc[stem_key].repeat(1, n_ch, 1, 1, 1) / n_ch

        self._unwrap().encoder.load_state_dict(enc, strict=True)  # guards architecture match
        self.print_to_log_file(f"Loaded {len(enc)} VoxTell encoder tensors from {path}")

    # ------------------------------- 2-stage schedule --------------------------- #
    def configure_optimizers(self, stage: str = "warmup_all"):
        params = self.network.parameters()
        if stage == "warmup_all":
            optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                        momentum=0.99, nesterov=True)
            lr_scheduler = Lin_incr_LRScheduler(optimizer, self.initial_lr, self.warmup_duration_whole_net)
        else:  # train -- reuse the warmup optimizer to keep momentum
            if self.training_stage == "warmup_all" and getattr(self, "optimizer", None) is not None:
                optimizer = self.optimizer
            else:
                optimizer = torch.optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                            momentum=0.99, nesterov=True)
            lr_scheduler = PolyLRScheduler_offset(optimizer, self.initial_lr, self.num_epochs,
                                                  self.warmup_duration_whole_net)
        self.training_stage = stage
        empty_cache(self.device)
        return optimizer, lr_scheduler

    def get_stage(self):
        return "warmup_all" if self.current_epoch < self.warmup_duration_whole_net else "train"

    def on_train_epoch_start(self):
        if self.current_epoch == 0:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("warmup_all")
        elif self.current_epoch == self.warmup_duration_whole_net:
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")
        super().on_train_epoch_start()

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        # initialize() always builds the warmup-stage scheduler; when resuming past the
        # warmup phase, swap in the train-stage (PolyLR) scheduler. The resumed optimizer
        # (and its momentum) is reused -- both stages train all parameters.
        if self.get_stage() == "train":
            self.optimizer, self.lr_scheduler = self.configure_optimizers("train")


class VoxTellTrainer_noMirroring(VoxTellTrainer):
    """Same as VoxTellTrainer but with mirroring augmentation disabled (needed for datasets
    whose labels distinguish left/right, e.g. Pengwin)."""

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
