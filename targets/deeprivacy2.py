"""
Rappresenta il target da sottoporre al test di sicurezza
"""

import sys
import types
import shutil
import os
from pathlib import Path
import torch
import numpy as np
from targets.base_target import BaseDeidentificationTarget

# 1. Definizione percorsi e import di base
repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository").resolve()
dp2_inner = repo_root / "deep_privacy2"

for p in [dp2_inner, repo_root]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# 2. Definizione funzione di risoluzione e patch globale
def _resolve_config_path(config_path):
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute() or not cfg_path.is_file():
        candidate = repo_root / cfg_path
        if candidate.is_file():
            return candidate
        candidate_inner = dp2_inner / cfg_path
        if candidate_inner.is_file():
            return candidate_inner
        matches = list(repo_root.glob(f"**/{cfg_path.name}"))
        if matches:
            return matches[0]
    return cfg_path

try:
    import dp2.utils
    _original_load_config = dp2.utils.load_config

    def _patched_load_config(config_path, *args, **kwargs):
        resolved = _resolve_config_path(config_path)
        return _original_load_config(resolved, *args, **kwargs)

    # Patch del modulo sorgente
    dp2.utils.load_config = _patched_load_config

    # Patch diretta nei moduli che hanno già importato `load_config`
    for mod_name in ["dp2.anonymizer.anonymizer", "dp2.anonymizer", "dp2.infer"]:
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], "load_config", _patched_load_config)
except ImportError:
    pass

# 3. Patch di PyTorch Hub per i pesi DSFD (offline mode)
weights_src = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-models/WIDERFace_DSFD_RES152.pth")

if weights_src.exists():
    cache_dir = Path("/root/.cache/torch/hub/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    expected_filename = "61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314"
    
    shutil.copy(weights_src, cache_dir / "WIDERFace_DSFD_RES152.pth")
    shutil.copy(weights_src, cache_dir / expected_filename)

    def _noop_download(url, dst, *args, **kwargs):
        if not Path(dst).exists():
            shutil.copy(weights_src, dst)

    torch.hub.download_url_to_file = _noop_download

# 4. Mock dipendenze opzionali / esterne
try:
    import motpy
except ModuleNotFoundError:
    motpy = types.ModuleType("motpy")
    motpy.Detection = object
    motpy.MultiObjectTracker = object
    sys.modules["motpy"] = motpy

try:
    import clip
except ModuleNotFoundError:
    clip = types.ModuleType("clip")
    clip.load = lambda *args, **kwargs: (None, None)
    clip.tokenize = lambda *args, **kwargs: None
    sys.modules["clip"] = clip

fake_cse_detector = types.ModuleType("dp2.detection.cse_mask_face_detector")
fake_cse_detector.CSeMaskFaceDetector = None
sys.modules["dp2.detection.cse_mask_face_detector"] = fake_cse_detector

fake_person_detector = types.ModuleType("dp2.detection.person_detector")
fake_person_detector.CSEPersonDetector = None
sys.modules["dp2.detection.person_detector"] = fake_person_detector

fake_dp2_utils_cse = types.ModuleType("dp2.utils.cse")
fake_dp2_utils_cse.from_E_to_vertex = lambda *args, **kwargs: None
sys.modules["dp2.utils.cse"] = fake_dp2_utils_cse

try:
    import densepose
except ModuleNotFoundError:
    class DummyDensePose:
        def __getattr__(self, item):
            return lambda *args, **kwargs: None

    densepose = types.ModuleType("densepose")
    sys.modules["densepose"] = densepose
    sys.modules["densepose.data"] = densepose
    sys.modules["densepose.data.utils"] = densepose
    sys.modules["densepose.modeling"] = densepose
    sys.modules["densepose.modeling.cse"] = densepose
    sys.modules["densepose.modeling.cse.utils"] = densepose
    sys.modules["densepose.structures"] = densepose


class DeepPrivacy2Target(BaseDeidentificationTarget):
    def __init__(self, config_path: str = "face_fdf128", models_dir: str = None):
        self.pipeline = self._load_pipeline(config_path, models_dir)

    def _load_pipeline(self, config_path, models_dir):
        from tops.config import LazyConfig, instantiate

        if config_path is None or config_path in ["fdf128", "stylegan_fdf128"]:
            config_path = "face_fdf128"

        cfg_path = _resolve_config_path(config_path)
        if not cfg_path.suffix:
            cfg_path = _resolve_config_path(f"{config_path}.py")

        if not cfg_path.exists():
            raise FileNotFoundError(f"Impossibile trovare la configurazione {config_path} in {repo_root}")

        cfg = LazyConfig.load(str(cfg_path))

        if models_dir:
            cfg.models_dir = str(Path(models_dir).resolve())

        if hasattr(cfg, "detector") and hasattr(cfg.detector, "name"):
            del cfg.detector.name

        # Configurazione percorsi scrivibili assoluti su /tmp
        writable_output_dir = Path("/tmp/outputs").resolve()
        writable_output_dir.mkdir(parents=True, exist_ok=True)
        
        cfg.output_dir = str(writable_output_dir)

        if hasattr(cfg, "detector"):
            cfg.detector.cache_directory = str(writable_output_dir / "face_detection_cache")

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo_root))
            
            # Assicura la patch all'interno di dp2.anonymizer.anonymizer se viene importato durante instantiate
            import dp2.anonymizer.anonymizer as anon_mod
            anon_mod.load_config = _patched_load_config

            if hasattr(cfg, "anonymizer"):
                pipeline = instantiate(cfg.anonymizer)
            elif hasattr(cfg, "generator"):
                from dp2.infer import build_trained_generator
                pipeline = build_trained_generator(cfg)
            else:
                pipeline = instantiate(cfg)
        finally:
            os.chdir(orig_cwd)

        return pipeline

    def process_image(self, image: torch.Tensor, *args, **kwargs) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        else:
            img_np = image

        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
