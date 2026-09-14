"""
Rappresenta il target da sottoporre al test di sicurezza
"""

import os
import sys
import types
import shutil
from pathlib import Path
import torch
import numpy as np
from targets.base_target import BaseDeidentificationTarget

# 1. Trova il file di pesi dal dataset Kaggle
weights_src = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-models/WIDERFace_DSFD_RES152.pth")

if weights_src.exists():
    # 2. Crea la cartella di cache standard di PyTorch Hub
    cache_dir = Path("/root/.cache/torch/hub/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Copia il file con il nome originale e con l'hash atteso
    expected_filename = "61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314"
    
    shutil.copy(weights_src, cache_dir / "WIDERFace_DSFD_RES152.pth")
    shutil.copy(weights_src, cache_dir / expected_filename)

    # 4. Patch di download_url_to_file per evitare qualsiasi richiesta URL
    def _noop_download(url, dst, *args, **kwargs):
        if not Path(dst).exists():
            shutil.copy(weights_src, dst)

    torch.hub.download_url_to_file = _noop_download
    
# 1. Mock di motpy se non presente
try:
    import motpy
except ModuleNotFoundError:
    motpy = types.ModuleType("motpy")
    motpy.Detection = object
    motpy.MultiObjectTracker = object
    sys.modules["motpy"] = motpy

# Mock di OpenAI CLIP per evitare la dipendenza
try:
    import clip
except ModuleNotFoundError:
    clip = types.ModuleType("clip")
    clip.load = lambda *args, **kwargs: (None, None)
    clip.tokenize = lambda *args, **kwargs: None
    sys.modules["clip"] = clip

# 2. Mock dei componenti CSE/Person di DeepPrivacy2 che dipendono da DensePose
fake_cse_detector = types.ModuleType("dp2.detection.cse_mask_face_detector")
fake_cse_detector.CSeMaskFaceDetector = None
sys.modules["dp2.detection.cse_mask_face_detector"] = fake_cse_detector

fake_person_detector = types.ModuleType("dp2.detection.person_detector")
fake_person_detector.CSEPersonDetector = None
sys.modules["dp2.detection.person_detector"] = fake_person_detector

fake_dp2_utils_cse = types.ModuleType("dp2.utils.cse")
fake_dp2_utils_cse.from_E_to_vertex = lambda *args, **kwargs: None
sys.modules["dp2.utils.cse"] = fake_dp2_utils_cse

# 3. Dummy fall-back universale per densepose se richiamato da file secondari
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
        import os
        repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository")
        dp2_inner = repo_root / "deep_privacy2"
        
        for path in [repo_root, dp2_inner]:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from tops.config import LazyConfig, instantiate

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

        if hasattr(cfg, "detector") and hasattr(cfg.detector, "name"):
            del cfg.detector.name

        # Reindirizzamento dell'output directory e della cache del detector su /tmp/outputs
        writable_output_dir = Path("/tmp/outputs")
        writable_output_dir.mkdir(parents=True, exist_ok=True)
        cfg.output_dir = str(writable_output_dir)

        if hasattr(cfg, "detector") and hasattr(cfg.detector, "cache_directory"):
            cfg.detector.cache_directory = str(writable_output_dir / "face_detection_cache")

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo_root))
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
