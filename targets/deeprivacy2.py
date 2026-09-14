"""
Rappresenta il target da sottoporre al test di sicurezza
"""


import sys
from pathlib import Path  # <-- Mancava questo import
import torch
import numpy as np
from targets.base_target import BaseDeidentificationTarget

# Adapter di DP2 nel framework
class DeepPrivacy2Target(BaseDeidentificationTarget):
    def __init__(self, config_path: str = "fdf128", models_dir: str = None):
        self.pipeline = self._load_pipeline(config_path, models_dir)

    def _load_pipeline(self, config_path, models_dir):
        repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository")
        dp2_inner = repo_root / "deep_privacy2"
        
        for path in [repo_root, dp2_inner]:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from dp2.infer import build_trained_generator
        import dp2.config as dp2_cfg

        if config_path is None:
            config_path = "fdf128"

        # Carica la configurazione tramite il modulo nativo dp2
        if hasattr(dp2_cfg, "load_config"):
            cfg = dp2_cfg.load_config(config_path)
        elif hasattr(dp2_cfg, "get_config"):
            cfg = dp2_cfg.get_config(config_path)
        else:
            from tops.config import LazyConfig
            cfg = LazyConfig.load(config_path)

        if models_dir:
            cfg.models_dir = models_dir

        return build_trained_generator(cfg)

    def process_image(self, image: torch.Tensor, *args, **kwargs) -> np.ndarray:
        """
        Riceve un tensore PyTorch e applica la pipeline DeepPrivacy2.
        """
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
