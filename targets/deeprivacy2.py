import sys
import os
import types
from pathlib import Path
import torch

# 0. MOCKING PREVENTIVO DI DENSEPOSE E DIPENDENZE OPZIONALI
class DummyModule(types.ModuleType):
    def __getattr__(self, name):
        return DummyModule(name)
    def __call__(self, *args, **kwargs):
        return self

sys.modules["densepose"] = DummyModule("densepose")
sys.modules["densepose.data"] = DummyModule("densepose.data")
sys.modules["densepose.data.utils"] = DummyModule("densepose.data.utils")
sys.modules["densepose.modeling"] = DummyModule("densepose.modeling")
sys.modules["densepose.modeling.cse"] = DummyModule("densepose.modeling.cse")
sys.modules["densepose.modeling.cse.utils"] = DummyModule("densepose.modeling.cse.utils")
sys.modules["densepose.structures"] = DummyModule("densepose.structures")

sys.modules["motpy"] = DummyModule("motpy")
sys.modules["clip"] = DummyModule("clip")
sys.modules["dp2.detection.cse_mask_face_detector"] = DummyModule("cse_mask")
sys.modules["dp2.detection.person_detector"] = DummyModule("person_detector")

fake_cse = types.ModuleType("dp2.utils.cse")
fake_cse.from_E_to_vertex = lambda *args, **kwargs: None
sys.modules["dp2.utils.cse"] = fake_cse


# 1. Configurazione dei percorsi di base su Kaggle
repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository").resolve()
dp2_inner = repo_root / "deep_privacy2"
models_dir = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-models")

for p in [dp2_inner, repo_root]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# 1.5 PATCH PREVENTIVA DI dp2.utils.load_config PER I PERCORSI RELATIVI
try:
    import dp2.utils as dp2_utils_mod
    orig_load_config = dp2_utils_mod.load_config
    def patched_load_config(config_path, *args, **kwargs):
        path_obj = Path(config_path)
        if not path_obj.is_file():
            candidate = repo_root / config_path
            if candidate.is_file():
                config_path = candidate
            else:
                candidate_inner = repo_root / "deep_privacy2" / config_path
                if candidate_inner.is_file():
                    config_path = candidate_inner
        return orig_load_config(config_path, *args, **kwargs)
    dp2_utils_mod.load_config = patched_load_config
    sys.modules["dp2.utils"] = dp2_utils_mod
except Exception as e:
    print(f"[WARNING] Impossibile applicare la patch preventiva a dp2.utils: {e}")


# 2. Patch robusta per DSFD e StyleGAN (Modalità 100% Offline)
dsfd_model_path = models_dir / "WIDERFace_DSFD_RES152.pth"
fdf_ckpt_path = models_dir / "stylegan_fdf128.ckpt"

try:
    import face_detection.dsfd.detect as dsfd_detect
    if dsfd_model_path.exists():
        dsfd_detect.model_url = str(dsfd_model_path)
        dsfd_detect.load_state_dict_from_url = lambda url, *args, **kwargs: torch.load(str(dsfd_model_path), map_location=kwargs.get("map_location", "cpu"))
except ImportError:
    pass

import tops
import tops.utils.file_util

def offline_load_file_or_url(path_or_url, map_location=None, md5sum=None):
    path_str = str(path_or_url)
    if "dsfd" in path_str.lower() or "widerface" in path_str.lower():
        if dsfd_model_path.exists():
            return str(dsfd_model_path)
            
    if fdf_ckpt_path.exists() and ("stylegan" in path_str.lower() or path_str.endswith(".ckpt") or "http" in path_str or not os.path.exists(path_str)):
        return str(fdf_ckpt_path)
            
    return path_str

tops.load_file_or_url = offline_load_file_or_url
try:
    tops.utils.file_util.load_file_or_url = offline_load_file_or_url
except AttributeError:
    pass


# 2.5 PATCH CRITICA A dp2.infer.load_generator_state PER GESTIRE IL CHECKPOINT LOCALE
try:
    import dp2.infer as dp2_infer
    def patched_load_generator_state(ckpt, G, ckpt_mapper=None):
        if isinstance(ckpt, (str, Path)):
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists() and fdf_ckpt_path.exists():
                ckpt_path = fdf_ckpt_path
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        
        # Estrazione sicura dello state dict indipendentemente dalle chiavi del dizionario
        state = None
        for key in ["EMA_generator", "running_average_generator", "generator", "state_dict"]:
            if isinstance(ckpt, dict) and key in ckpt:
                state = ckpt[key]
                break
        if state is None and isinstance(ckpt, dict):
            state = ckpt # Fallback sull'intero dizionario se non trova chiavi note
            
        if hasattr(G, "load_state_dict"):
            try:
                G.load_state_dict(state, strict=False)
            except Exception:
                # Se fallisce, prova a caricare direttamente lo stato o rimappa
                pass
        return dp2_infer.orig_load_generator_state(ckpt, G, ckpt_mapper) if hasattr(dp2_infer, "orig_load_generator_state") else None

    dp2_infer.orig_load_generator_state = dp2_infer.load_generator_state
    dp2_infer.load_generator_state = patched_load_generator_state
except Exception as e:
    print(f"[WARNING] Impossibile applicare la patch a dp2.infer: {e}")


# 3. Classe Target che istanzia l'Anonymizer nativo di DeepPrivacy2
from targets.base_target import BaseDeidentificationTarget

class DeepPrivacy2Target(BaseDeidentificationTarget):
    def __init__(self, config_path: str = None, models_dir: str = None):
        self.pipeline = self._load_pipeline()
        
        self.face_G = None
        if hasattr(self.pipeline, "generators"):
            # Cerca nel dizionario dei generatori della pipeline di DeepPrivacy2
            for k, gen in self.pipeline.generators.items():
                if "face" in str(k).lower() or "generator" in str(type(gen)).lower():
                    self.face_G = gen
                    break
            if self.face_G is None and len(self.pipeline.generators) > 0:
                self.face_G = list(self.pipeline.generators.values())[0]
                
    def _load_pipeline(self):
        from tops.config import LazyConfig, instantiate

        possible_paths = [
            repo_root / "configs" / "anonymizers" / "face.py",
            repo_root / "configs" / "anonymizers" / "face_fdf128.py",
            repo_root / "deep_privacy2" / "configs" / "anonymizers" / "face.py",
            repo_root / "deep_privacy2" / "configs" / "anonymizers" / "face_fdf128.py"
        ]

        face_cfg_path = None
        for p in possible_paths:
            if p.exists():
                face_cfg_path = p
                break

        if not face_cfg_path:
            for p in repo_root.glob("**/face.py"):
                if "gradio_demos" not in str(p) and "demos" not in str(p):
                    face_cfg_path = p
                    break

        if not face_cfg_path or not face_cfg_path.exists():
            raise FileNotFoundError(f"Impossibile trovare un file di configurazione valido per l'anonymizer in {repo_root}")

        print(f"[INFO] Caricamento configurazione anonymizer da: {face_cfg_path}")

        orig_cwd = os.getcwd()
        try:
            os.chdir("/tmp")
            sys.path_importer_cache.clear()

            cfg = LazyConfig.load(str(face_cfg_path))
            
            writable_output_dir = Path("/tmp/outputs").resolve()
            writable_output_dir.mkdir(parents=True, exist_ok=True)
            cfg.output_dir = str(writable_output_dir)

            anonymizer_instance = instantiate(cfg.anonymizer)
            return anonymizer_instance

        finally:
            os.chdir(orig_cwd)

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
        try:
            if hasattr(self.pipeline, "__call__"):
                res = self.pipeline(img_batch.float().to(device))
                anonymized_img = res[0] if isinstance(res, (list, tuple)) else res
            else:
                anonymized_img = img_batch
        except Exception as e:
            print(f"[WARNING] Errore durante il processamento immagine con anonymizer: {e}")
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
