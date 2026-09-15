import sys
import os
import types
from pathlib import Path
import torch

# 0. MOCKING PREVENTIVO DI DENSEPOSE (Risolve il blocco corrente)
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

# Altri mock sicuri
sys.modules["motpy"] = DummyModule("motpy")
sys.modules["clip"] = DummyModule("clip")
sys.modules["dp2.detection.cse_mask_face_detector"] = DummyModule("cse_mask")
sys.modules["dp2.detection.person_detector"] = DummyModule("person_detector")

# Funzione mock per cse.py
fake_cse = types.ModuleType("dp2.utils.cse")
fake_cse.from_E_to_vertex = lambda *args, **kwargs: None
sys.modules["dp2.utils.cse"] = fake_cse


# 1. Configurazione dei percorsi di base
repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository").resolve()
dp2_inner = repo_root / "deep_privacy2"
models_dir = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-models")

for p in [dp2_inner, repo_root]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# 2. Patch robusta per DSFD e StyleGAN basata sul tuo vecchio codice funzionante
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
            
    if path_str.startswith("http://") or path_str.startswith("https://") or not os.path.exists(path_str):
        if fdf_ckpt_path.exists():
            return str(fdf_ckpt_path)
            
    return path_str

tops.load_file_or_url = offline_load_file_or_url
try:
    tops.utils.file_util.load_file_or_url = offline_load_file_or_url
except AttributeError:
    pass


# 3. Classe Target che istanzia l'Anonymizer nativo
from targets.base_target import BaseDeidentificationTarget

class DeepPrivacy2Target(BaseDeidentificationTarget):
    def __init__(self, config_path: str = None, models_dir: str = None):
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self):
        from tops.config import LazyConfig, instantiate

        face_cfg_path = repo_root / "configs/anonymizers/face.py"
        if not face_cfg_path.exists():
            face_cfg_path = repo_root / "configs/anonymizers/face_fdf128.py"
        if not face_cfg_path.exists():
            matches = list(repo_root.glob("**/face.py")) + list(repo_root.glob("**/face_fdf128.py"))
            if matches:
                face_cfg_path = matches[0]

        orig_cwd = os.getcwd()
        try:
            os.chdir(str(repo_root))
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
