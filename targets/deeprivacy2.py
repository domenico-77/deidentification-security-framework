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
    resolved = _resolve_config_path(config_path)
    if _original_load_config is not None:
        return _original_load_config(resolved, *args, **kwargs)

    # Fallback se la funzione originale non è catturata
    from tops.config import LazyConfig
    return LazyConfig.load(str(resolved))


# Tentativo di patch dei moduli già importati o disponibili
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
    
    # Gestione flessibile delle varie chiavi usate nei checkpoint di DeepPrivacy2
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
        state = ckpt  # Fallback se il file contiene direttamente lo state_dict

    if ckpt_mapper is not None:
        state = ckpt_mapper(state)
        
    generator.load_state_dict(state, strict=False)


# Applicazione della patch al modulo infer di DP2
try:
    import dp2.infer
    dp2.infer.load_generator_state = _patched_load_generator_state
except ImportError:
    pass
# 5. Classe Target per la De-identificazione
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

        # Configurazione directory di output scrivibile
        writable_output_dir = Path("/tmp/outputs").resolve()
        writable_output_dir.mkdir(parents=True, exist_ok=True)
        
        cfg.output_dir = str(writable_output_dir)

        if hasattr(cfg, "detector"):
            cfg.detector.cache_directory = str(writable_output_dir / "face_detection_cache")

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo_root))
            
            # Applicazione protetta della patch nel contesto di istanziazione
            try:
                import dp2.anonymizer.anonymizer as anon_mod
                anon_mod.load_config = _patched_load_config
            except ImportError:
                pass

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

    def process_image(self, image_tensor):
        """
        Processa un tensore immagine tramite la pipeline di DeepPrivacy2.
        """
        img = image_tensor.detach().clone()
        if img.max() <= 1.0:
            img = img * 255.0
        
        # Converte in formato uint8 CPU (H, W, C) come numpy array, spesso richiesto dai wrapper di DP2
        img_np = img.permute(1, 2, 0).byte().cpu().numpy()

        anonymized_img = None
        
        # 1. Tentativo tramite i metodi standard di elaborazione immagine della pipeline
        for method_name in ["anonymize_image", "anonymize", "process"]:
            if hasattr(self.pipeline, method_name):
                try:
                    func = getattr(self.pipeline, method_name)
                    anonymized_img = func(img_np)
                    break
                except Exception:
                    continue

        # 2. Se i metodi falliscono, proviamo a passare l'input come lista o dizionario a __call__
        if anonymized_img is None and hasattr(self.pipeline, "__call__"):
            try:
                # Alcune pipeline DP2 accettano liste di numpy arrays o tensori batch
                inputs = [img_np]
                res = self.pipeline(inputs)
                anonymized_img = res[0] if isinstance(res, (list, tuple)) else res
            except Exception:
                try:
                    # Fallback con tensore batch torch.uint8
                    batch_tensor = img.byte().unsqueeze(0)
                    res = self.pipeline(batch_tensor)
                    anonymized_img = res[0] if isinstance(res, torch.Tensor) and res.dim() == 4 else res
                except Exception as e:
                    raise RuntimeError(f"Tutti i tentativi di chiamata alla pipeline DeepPrivacy2 sono falliti: {e}")

        if anonymized_img is None:
            raise RuntimeError("Impossibile completare l'anonimizzazione tramite la pipeline DeepPrivacy2.")

        # Conversione finale del risultato in tensore float normalizzato [0, 1] con formato (C, H, W)
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

        return out_tensor.to(image_tensor.device)
