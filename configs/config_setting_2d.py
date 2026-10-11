"""RGB binary StructDiffSeg settings; optimizer/loss mirror ACDC defaults."""

from dataclasses import dataclass
from pathlib import Path

from models.SPM5121.vit_seg_configs import get_r50_b16_config
from utils import CeDiceLoss


@dataclass
class TrainingConfig2D:
    datasets_name: str
    data_path: str = None
    list_dir: str = None
    batch_size: int = 1
    epochs: int = 100
    num_workers: int = 0
    seed: int = 42
    input_channels: int = 3
    num_classes: int = 2
    input_size_h: int = 256
    input_size_w: int = 256
    ddim_steps: int = 10
    amp: bool = False
    print_interval: int = 10
    opt: str = 'AdamW'
    lr: float = 0.001
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    sch: str = 'CosineAnnealingLR'
    eta_min: float = 0.00001
    last_epoch: int = -1

    def __post_init__(self):
        if self.datasets_name not in ('glas', 'cvc'):
            raise ValueError('datasets_name must be glas or cvc')
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError('Invalid batch_size, epochs or num_workers')
        if (self.input_channels, self.num_classes,
                self.input_size_h, self.input_size_w) != (3, 2, 256, 256):
            raise ValueError('This entry point requires RGB, 2 classes and 256x256 inputs')
        if self.amp:
            raise ValueError('AMP is disabled: the existing AMP engine path is not diffusion training')
        root = Path(__file__).resolve().parents[2]
        self.data_path = str(Path(self.data_path) if self.data_path else root / self.datasets_name)
        self.list_dir = str(Path(self.list_dir) if self.list_dir else Path(self.data_path))
        self.T_max = self.epochs

    @property
    def model_config(self):
        return {'input_channels': self.input_channels, 'num_classes': self.num_classes}

    def make_criterion(self):
        return CeDiceLoss(self.num_classes, [1, 1])

    def make_spm_config(self):
        # Use the same configuration factory and patch layout as train_acdc.py.
        config = get_r50_b16_config()
        config.n_classes = self.num_classes
        config.n_skip = 3
        config.batch_size = self.batch_size
        config.patches.grid = (self.input_size_h // 16, self.input_size_w // 16)
        config.n_patches = config.patches.grid[0] * config.patches.grid[1]
        config.h, config.w = config.patches.grid
        return config
