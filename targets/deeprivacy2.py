"""
Rappresenta il target da sottoporre al test di sicurezza
"""


import sys
import types
from pathlib import Path  # <-- Mancava questo import
import torch
import numpy as np
from targets.base_target import BaseDeidentificationTarget

# Mock di motpy se non presente
try:
    import motpy
except ModuleNotFoundError:
    motpy = types.ModuleType("motpy")
    motpy.Detection = object
    motpy.MultiObjectTracker = object
    sys.modules["motpy"] = motpy

# Mock di densepose per evitare ModuleNotFoundError importando dp2.utils
try:
    import densepose
except ModuleNotFoundError:
    densepose = types.ModuleType("densepose")
    modeling = types.ModuleType("densepose.modeling")
    cse = types.ModuleType("densepose.modeling.cse")
    cse_utils = types.ModuleType("densepose.modeling.cse.utils")
    
    # Mock della funzione usata in dp2/utils/cse.py
    cse_utils.get_closest_vertices_mask_from_ES = lambda *args, **kwargs: None
    
    cse.utils = cse_utils
    modeling.cse = cse
    densepose.modeling = modeling
    
    sys.modules["densepose"] = densepose
    sys.modules["densepose.modeling"] = modeling
    sys.modules["densepose.modeling.cse"] = cse
    sys.modules["densepose.modeling.cse.utils"] = cse_utils


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

        if config_path is None or config_path in ["fdf128", "stylegan_fdf128"]:
            config_path = "face_fdf128"

        cfg_path = Path(config_path)
        if not cfg_path.suffix:
            cfg_path = Path(f"{config_path}.py")

        if not cfg_path.exists():
            candidates = list(repo_root.glob(f"**/configs/**/{cfg_path.name}"))
            if not candidates:
                candidates = list(repo_root.glob(f"**/{cfg_path.name}"))
            if candidates:
                cfg_path = candidates[0]
            else:
                raise FileNotFoundError(f"Impossibile trovare la configurazione {cfg_path.name} in {repo_root}")

        cfg = LazyConfig.load(str(cfg_path))

        if models_dir:
            cfg.models_dir = models_dir

        return build_trained_generator(cfg)

    def process_image(self, image: torch.Tensor, *args, **kwargs) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        else:
            img_np = image

        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
