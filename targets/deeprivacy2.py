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
    def __init__(self, config_path: str = None, models_dir: str = None):
        # Inizializzazione e caricamento offline dei modelli DSFD e StyleGAN2
        self.pipeline = self._load_pipeline(config_path, models_dir)

    def _load_pipeline(self, config_path, models_dir):
    # La cartella MADRE che contiene il package 'deep_privacy2'
    repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository")
    if repo_root.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Ora Python troverà la cartella 'deep_privacy2' dentro 'repo_root'
    from deep_privacy2 import build_deidentifier
    return build_deidentifier("fdf128", models_dir=models_dir)

    def process_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Riceve tensore PyTorch e applica DeepPrivacy2.
        Se il detector è ingannato, restituisce l'immagine originale inalterata.
        """
        img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
