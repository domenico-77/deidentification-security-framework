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
        from tops.config import LazyConfig

        if config_path is None:
            config_path = "fdf128"

        # Risoluzione del percorso di configurazione
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            # Cerca tra i file di configurazione presenti nel repository DeepPrivacy2
            possible_path = dp2_inner / "configs" / f"{config_path}.py"
            if possible_path.exists():
                cfg_path = possible_path

        # Caricamento via LazyConfig.load()
        cfg = LazyConfig.load(str(cfg_path))

        if models_dir:
            cfg.models_dir = models_dir

        return build_trained_generator(cfg)

    def process_image(self, image: torch.Tensor, *args, **kwargs) -> np.ndarray:
        """
        Riceve un tensore PyTorch e applica la pipeline DeepPrivacy2.
        """
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        else:
            img_np = image

        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
