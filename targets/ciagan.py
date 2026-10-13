import sys
import os
from pathlib import Path
import torch
import numpy as np

# 1. Configurazione dinamica dei percorsi indipendenti dall'utente
KAGGLE_INPUT = Path("/kaggle/input")

if KAGGLE_INPUT.exists():
    repo_matches = list(KAGGLE_INPUT.glob("**/ciagan-repository"))
    models_matches = list(KAGGLE_INPUT.glob("**/ciagan-models"))
    
    if repo_matches:
        ciagan_repo_root = (repo_matches[0] / "ciagan").resolve()
    else:
        ciagan_repo_root = Path("/kaggle/input/ciagan-repository/ciagan").resolve()
        
    if models_matches:
        ciagan_models_dir = models_matches[0].resolve()
    else:
        ciagan_models_dir = Path("/kaggle/input/ciagan-models").resolve()
else:
    base_local = Path("./data")
    ciagan_repo_root = (base_local / "ciagan-repository/ciagan").resolve()
    ciagan_models_dir = (base_local / "ciagan-models").resolve()

ciagan_source_dir = ciagan_repo_root / "source"

# Aggiunta dei percorsi al sys.path
for p in [ciagan_repo_root, ciagan_source_dir]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from targets.base_target import BaseDeidentificationTarget

class CIAGANTarget(BaseDeidentificationTarget):
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.detector = self._load_detector()

    def _load_model(self):
        """Inizializza l'architettura del generatore di CIAGAN da arch_unet_flex e carica i pesi."""
        try:
            from source.arch.arch_unet_flex import UNet as Generator
        except ImportError:
            try:
                from arch.arch_unet_flex import UNet as Generator
            except ImportError as e:
                raise ImportError(f"[ERRORE] Impossibile importare l'architettura U-Net di CIAGAN: {e}")

        # Inizializza il modello (specificando i parametri di default se richiesti dall'architettura)
        model = Generator()
        
        checkpoint_path = ciagan_models_dir / "modelG_ciagan.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"[ERRORE] Checkpoint non trovato in: {checkpoint_path}")

        print(f"[INFO] Caricamento pesi CIAGAN da: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "generator", "G"]:
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _load_detector(self):
        """Carica il detector DSFD per la compatibilità con il BenchmarkRunner."""
        try:
            from face_detection import FD
            dsfd_path = None
            if KAGGLE_INPUT.exists():
                matches = list(KAGGLE_INPUT.glob("**/WIDERFace_DSFD_RES152.pth"))
                if matches:
                    dsfd_path = matches[0]
            
            if dsfd_path and dsfd_path.exists():
                import face_detection.dsfd.detect as dsfd_detect
                dsfd_detect.model_url = str(dsfd_path)
                dsfd_detect.load_state_dict_from_url = lambda url, *args, **kwargs: torch.load(str(dsfd_path), map_location=kwargs.get("map_location", "cpu"))
            
            detector = FD("dsfd", device=self.device)
            return detector
        except Exception as e:
            print(f"[WARNING] Impossibile inizializzare il detector DSFD in CIAGANTarget: {e}")
            return None

    def process_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processa l'immagine tensoriale attraverso la pipeline di CIAGAN
        rispettando l'interfaccia BaseDeidentificationTarget.
        """
        device = self.device
        img = image_tensor.detach().clone()
        
        if img.max() <= 1.0:
            img = img * 255.0

        if img.dim() == 3:
            img_batch = img.unsqueeze(0)
        else:
            img_batch = img

        img_batch = img_batch.float().to(device)

        with torch.no_grad():
            normalized_input = (img_batch / 127.5) - 1.0
            output_tensor = self.model(normalized_input)
            output_tensor = (output_tensor + 1.0) / 2.0
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        if output_tensor.dim() == 4:
            output_tensor = output_tensor.squeeze(0)

        return output_tensor.to(device)
