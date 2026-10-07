import torch
import torch.nn as nn
from ultralytics import YOLO
from detectors.base_detector import BaseDetector

class YoloFaceDetector(BaseDetector):
    """
    Wrapper per YOLOv8-face ottimizzato per il calcolo dei gradienti 
    tramite estrazione di feature intermedie (feature-level attack).
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        
        try:
            self.yolo_instance = YOLO(weights_path)
            # Estraiamo il modello PyTorch sottostante
            self.net = self.yolo_instance.model.to(self.device).eval()
            
            # Salviamo una copia delle feature pulite originali per il calcolo della loss di divergenza
            self.target_features = None
            self._register_hook()
            
        except Exception as e:
            print(f"[ERROR] Impossibile caricare il modello YOLO da {weights_path}: {e}")
            raise e

    def _register_hook(self):
        """
        Registra un forward hook su un livello intermedio (es. l'ultimo blocco del neck/backbone)
        per catturare i tensori intermedi ed evitare il blocco del gradiente della testa di detection.
        """
        # In YOLOv8, self.net.model[-2] o [-3] è solitamente un blocco di feature prima della Detect head (-1)
        target_layer = self.net.model[-2] 
        
        def hook_fn(module, input, output):
            self.extracted_features = output

        target_layer.register_forward_hook(hook_fn)

    def to(self, device):
        self.device = device
        if hasattr(self, 'net') and self.net is not None:
            self.net = self.net.to(device)
        return self

    def zero_grad(self):
        if hasattr(self, 'net') and self.net is not None:
            self.net.zero_grad()

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        """Restituisce il numero di volti rilevati tramite predizione YOLO standard."""
        with torch.no_grad():
            results = self.yolo_instance(image_tensor, verbose=False)
            count = 0
            for r in results:
                if r.boxes is not None:
                    count += len(r.boxes)
            return count

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calcola la loss di attacco massimizzando la distanza o perturbando 
        le feature intermedie estratte dalla rete anziché usare l'output finale.
        """
        if image_tensor.dim() == 5:
            b, n, c, h, w = image_tensor.shape
            image_tensor = image_tensor.view(b * n, c, h, w)
            
        # Eseguiamo il forward pass (attiverà l'hook sullo strato intermedio)
        _ = self.net(image_tensor)
        
        # Se non abbiamo ancora salvato le feature originali (pulite) per il confronto, le memorizziamo
        if not hasattr(self, 'clean_features') or self.clean_features is None:
            self.clean_features = self.extracted_features.detach()

        # Loss di feature divergence (es. MSE loss invertita o L2 norm massimizzata)
        # Vogliamo spingere le feature dell'immagine avversaria lontano da quelle originali
        loss = -torch.nn.functional.mse_loss(self.extracted_features, self.clean_features)
        
        return loss

    def __call__(self, x):
        if x.dim() == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        return self.net(x)
