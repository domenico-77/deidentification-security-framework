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
    def __init__(self, config_path: str = "face_fdf128", models_dir: str = None):
        self.pipeline = self._load_pipeline(config_path, models_dir)

    def _load_pipeline(self, config_path, models_dir):
        repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository")
        dp2_inner = repo_root / "deep_privacy2"
        
        for path in [repo_root, dp2_inner]:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from dp2.infer import build_trained_generator
        from tops.config import LazyConfig

        if config_path is None or config_path == "fdf128":
            config_path = "face_fdf128"

        cfg_path = Path(config_path)
        if not cfg_path.suffix:
            cfg_path = Path(f"{config_path}.py")

        # Cerca ricorsivamente il file di configurazione tra gli anonymizers/configs
        if not cfg_path.exists():
            candidates = list(repo_root.glob(f"**/configs/**/{cfg_path.name}"))
            if not candidates:
                candidates = list(repo_root.glob(f"**/{cfg_path.name}"))
            
            if candidates:
                cfg_path = candidates[0]
            else:
                raise FileNotFoundError(f"Impossibile trovare la configurazione {cfg_path.name} in {repo_root}")

        # Caricamento via LazyConfig
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
