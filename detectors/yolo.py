import torch
import torch.nn as nn
from ultralytics import YOLO
from detectors.base_detector import BaseDetector

class YoloFaceDetector(BaseDetector):
    """
    Wrapper per YOLOv8-face da utilizzare come modello surrogato 
    per i Transfer Attack, implementando l'interfaccia BaseDetector.
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        
        try:
            # Caricamento diretto del modello dai pesi locali specificati
            self.yolo_instance = YOLO(weights_path)
            # Estraiamo il modulo PyTorch sottostante per il calcolo dei gradienti
            self.net = self.yolo_instance.model.to(self.device).eval()
        except Exception as e:
            print(f"[ERROR] Impossibile caricare il modello YOLO da {weights_path}: {e}")
            raise e

    def to(self, device):
        """Supporta il metodo .to() richiesto dal framework di attacco."""
        self.device = device
        if hasattr(self, 'net') and self.net is not None:
            self.net = self.net.to(device)
        return self

    def zero_grad(self):
        """Supporta il metodo .zero_grad() richiesto dal framework di attacco."""
        if hasattr(self, 'net') and self.net is not None:
            self.net.zero_grad()

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        """Restituisce il numero di volti rilevati tramite predizione YOLO."""
        with torch.no_grad():
            results = self.yolo_instance(image_tensor, verbose=False)
            count = 0
            for r in results:
                if r.boxes is not None:
                    count += len(r.boxes)
            return count

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Calcola la loss basata sull'output del modello sottostante per il gradiente."""
        if image_tensor.dim() == 5:
            b, n, c, h, w = image_tensor.shape
            image_tensor = image_tensor.view(b * n, c, h, w)
            
        outputs = self.net(image_tensor)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    loss = loss + torch.relu(out).sum()
        elif isinstance(outputs, torch.Tensor):
            loss = torch.relu(outputs).sum()
            
        return loss

    def __call__(self, x):
        """Forward pass diretto sul modello PyTorch interno per l'attacco."""
        if x.dim() == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        return self.net(x)
