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
        repo_root = Path("/kaggle/input/datasets/domenicovicenti/deep-privacy2-repository")
        
        if repo_root.exists() and str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    
        # Importa direttamente le funzioni di caricamento dal modulo di inferenza
        from deep_privacy2.config import load_config
        from deep_privacy2.dp2.infer import build_trained_generator
    
        # Carica la configurazione del modello (es. fdf128)
        cfg = load_config("fdf128", models_dir=models_dir)
        
        # Inizializza il generatore addestrato
        generator = build_trained_generator(cfg)
        return generator

    def process_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Riceve tensore PyTorch e applica DeepPrivacy2.
        Se il detector è ingannato, restituisce l'immagine originale inalterata.
        """
        img_np = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        anonymized_img = self.pipeline.anonymize_image(img_np)
        return anonymized_img
