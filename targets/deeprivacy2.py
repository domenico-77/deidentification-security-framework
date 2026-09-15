import sys
import types
import shutil
import os
from pathlib import Path
import torch
import numpy as np
from targets.base_target import BaseDeidentificationTarget

# 1. Definizione percorsi di base e sys.path
repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository").resolve()
dp2_inner = repo_root / "deep_privacy2"

for p in [dp2_inner, repo_root]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# 2. Definizione GLOBALE delle funzioni di risoluzione e patch
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


_original_load_config = None


def _patched_load_config(config_path, *args, **kwargs):
    if config_path and ("stylegan_fdf128" in str(config_path) or "fdf128" in str(config_path)):
        config_path = "configs/fdf/stylegan.py"
        
    resolved = _resolve_config_path(config_path)
    if _original_load_config is not None:
        return _original_load_config(resolved, *args, **kwargs)

    from tops.config import LazyConfig
    return LazyConfig.load(str(resolved))


try:
    import dp2.utils
    _original_load_config = dp2.utils.load_config
    dp2.utils.load_config = _patched_load_config

    for mod_name in ["dp2.anonymizer.anonymizer", "dp2.anonymizer", "dp2.infer"]:
        if mod_name in sys.modules:
            setattr(sys.modules[mod_name], "load_config", _patched_load_config)
except ImportError:
    pass


# 3. Patch per PyTorch Hub (modalità offline per pesi DSFD)
weights_dir = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-models")
dsfd_weights = weights_dir / "WIDERFace_DSFD_RES152.pth"

if dsfd_weights.exists():
    cache_dir = Path("/root/.cache/torch/hub/checkpoints")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    expected_filename = "61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314"
    
    shutil.copy(dsfd_weights, cache_dir / "WIDERFace_DSFD_RES152.pth")
    shutil.copy(dsfd_weights, cache_dir / expected_filename)

    def _noop_download(url, dst, *args, **kwargs):
        if not Path(dst).exists():
            shutil.copy(dsfd_weights, dst)

    torch.hub.download_url_to_file = _noop_download


# 3.5. Patch mirata per il caricamento offline dei pesi (DSFD vs StyleGAN)
try:
    import tops
    _original_load_file_or_url = tops.load_file_or_url

    def _patched_load_file_or_url(file_url, *args, **kwargs):
        url_str = str(file_url).lower()
        
        all_candidates = []
        for d in [weights_dir, repo_root / "models"]:
            if d.exists():
                all_candidates.extend(list(d.glob("**/*.pth")) + list(d.glob("**/*.ckpt")))

        if "dsfd" in url_str or "widerface" in url_str:
            if dsfd_weights.exists():
                return str(dsfd_weights)
        
        generator_candidates = [c for c in all_candidates if "dsfd" not in c.name.lower() and "widerface" not in c.name.lower()]
        if generator_candidates:
            return str(generator_candidates[0])
                        
        return _original_load_file_or_url(file_url, *args, **kwargs)

    tops.load_file_or_url = _patched_load_file_or_url
except ImportError:
    pass


# 4. Mocking dipendenze esterne opzionali o mancanti nel container Kaggle
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

def _patched_load_generator_state(ckpt, generator, ckpt_mapper=None):
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    
    if "generator" in ckpt:
        state = ckpt["generator"]
    elif "EMA_generator" in ckpt:
        state = ckpt["EMA_generator"]
    elif "running_average_generator" in ckpt:
        state = ckpt["running_average_generator"]
    elif "generator_state" in ckpt:
        state = ckpt["generator_state"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt  

    if ckpt_mapper is not None:
        state = ckpt_mapper(state)
        
    generator.load_state_dict(state, strict=False)


try:
    import dp2.infer
    dp2.infer.load_generator_state = _patched_load_generator_state
except ImportError:
    pass


# Wrapper per garantire che la pipeline esponga sia .generator che .detector per PGDAttack
class DeepPrivacy2PipelineWrapper:
    def __init__(self, generator, detector=None):
        self.generator = generator
        self.detector = detector
        if self.detector is None:
            # Inizializza un detector di fallback compatibile se non presente
            try:
                from detectors.dsfd import DSFDDetector
                # Crea un'istanza leggera o usa il detector DSFD standard del framework
                self.detector = types.SimpleNamespace()
                from face_detection import DSFD
                self.detector.face_detector = DSFD()
                self.detector.face_mean = torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)
            except Exception:
                pass


# 5. Classe Target per la De-identificazione
class DeepPrivacy2Target(BaseDeidentificationTarget):
    def __init__(self, config_path: str = "configs/fdf/stylegan.py", models_dir: str = None):
        self.pipeline = self._load_pipeline(config_path, models_dir)

    def _load_pipeline(self, config_path, models_dir):
        from tops.config import LazyConfig, instantiate

        if config_path is None or config_path in ["fdf128", "stylegan_fdf128", "face_fdf128"]:
            config_path = "configs/fdf/stylegan.py"

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

        writable_output_dir = Path("/tmp/outputs").resolve()
        writable_output_dir.mkdir(parents=True, exist_ok=True)
        
        cfg.output_dir = str(writable_output_dir)

        if hasattr(cfg, "detector"):
            cfg.detector.cache_directory = str(writable_output_dir / "face_detection_cache")

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo_root))
            
            try:
                import dp2.anonymizer.anonymizer as anon_mod
                anon_mod.load_config = _patched_load_config
            except ImportError:
                pass

            if hasattr(cfg, "anonymizer"):
                raw_pipeline = instantiate(cfg.anonymizer)
            elif hasattr(cfg, "generator"):
                from dp2.infer import build_trained_generator
                raw_pipeline = build_trained_generator(cfg)
            else:
                raw_pipeline = instantiate(cfg)
        finally:
            os.chdir(orig_cwd)

        # Creazione del wrapper per esporre correttamente detector e generator a PGDAttack
        detector_obj = getattr(raw_pipeline, "detector", None)
        if detector_obj is None and hasattr(raw_pipeline, "detectors"):
            from dp2.anonymizer.anonymizer import FaceDetection
            detector_obj = raw_pipeline.detectors.get(FaceDetection, None)

        generator_obj = getattr(raw_pipeline, "generator", raw_pipeline)

        pipeline_wrapper = DeepPrivacy2PipelineWrapper(generator=generator_obj, detector=detector_obj)
        return pipeline_wrapper

    def process_image(self, image_tensor):
        device = image_tensor.device
        img = image_tensor.detach().clone()
        if img.max() <= 1.0:
            img = img * 255.0
        
        if img.dim() == 3:
            img_uint8 = img.byte()
            img_batch = img_uint8.unsqueeze(0)
        else:
            img_batch = img.byte()
            img_uint8 = img_batch.squeeze(0)

        anonymized_img = None
        gen_to_use = getattr(self.pipeline, "generator", self.pipeline)

        try:
            detector_obj = getattr(self.pipeline, "detector", None)
            if detector_obj is not None and hasattr(detector_obj, "detect_faces"):
                boxes, _ = detector_obj.detect_faces(img_batch)
                if boxes is not None and len(boxes) > 0:
                    gen_output = gen_to_use(img_batch.float().to(device), boxes)
                    anonymized_img = gen_output[0] if isinstance(gen_output, (list, tuple)) else gen_output
                else:
                    anonymized_img = img_batch
            else:
                anonymized_img = img_batch
        except Exception:
            anonymized_img = img_batch

        if isinstance(anonymized_img, torch.Tensor):
            out_tensor = anonymized_img.detach().cpu()
            if out_tensor.dim() == 4:
                out_tensor = out_tensor.squeeze(0)
        else:
            import numpy as np
            if isinstance(anonymized_img, tuple):
                anonymized_img = anonymized_img[0]
            out_tensor = torch.from_numpy(np.array(anonymized_img))
            if out_tensor.dim() == 3 and out_tensor.shape[2] == 3:
                out_tensor = out_tensor.permute(2, 0, 1)

        if out_tensor.dtype != torch.float32:
            out_tensor = out_tensor.float()

        if out_tensor.max() > 1.0:
            out_tensor = out_tensor / 255.0

        return out_tensor.to(device)
