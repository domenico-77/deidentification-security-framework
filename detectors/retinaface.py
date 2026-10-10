import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector

class RetinaFaceDetector(BaseDetector):
    """
    Wrapper per RetinaFace (MobileNet0.25) da utilizzare come modello surrogato 
    per i Transfer Attack, implementando l'interfaccia BaseDetector.
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.net = self._load_retinaface(weights_path)

    def _load_retinaface(self, weights_path: str) -> nn.Module:
        """Inizializza l'architettura e carica i pesi."""
        try:
            # Tentativo di importazione da librerie esterne comuni o pip package
            try:
                from retinaface.net import RetinaFace
            except ImportError:
                try:
                    from models.retinaface.retinaface import RetinaFace
                except ImportError:
                    # Definizione minimale o caricamento tramite torchvision se disponibile
                    import torchvision.models as models
                    # Fallback robusto se il pacchetto retinaface non è strutturato come modulo locale
                    raise ImportError("Impossibile trovare la definizione della classe RetinaFace. Assicurati che il modulo sia installato o presente.")

            cfg = {
                'name': 'mobilenet0.25',
                'min_sizes': [[16, 32], [64, 128], [256, 512]],
                'steps': [8, 16, 32],
                'variance': [0.1, 0.2],
                'clip': False,
                'loc_weight': 2.0,
                'cls_weight': 1.0,
                'landm_weight': 1.0,
                'pretrain': False
            }
            net = RetinaFace(cfg=cfg, phase='test')
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            net.load_state_dict(new_state_dict, strict=False)
            
            return net.to(self.device).eval()
        except Exception as e:
            print(f"[ERROR] Impossibile caricare RetinaFace: {e}")
            raise e

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        """Restituisce il numero di volti rilevati."""
        with torch.no_grad():
            out = self.net(image_tensor)
            if isinstance(out, tuple) and len(out) >= 2:
                conf = out[1]
                if conf.dim() == 3:
                    valid = (conf[:, :, 1] > 0.5).sum().item()
                    return int(valid)
        return 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Calcola la loss da massimizzare per far evadere il detector."""
        out = self.net(image_tensor)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if isinstance(out, tuple):
            for t in out:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    loss = loss + torch.relu(t).sum()
        elif isinstance(out, torch.Tensor):
            loss = torch.relu(out).sum()
            
        return loss

    def __call__(self, x):
        """Permette di invocare direttamente il modello surrogato per il calcolo dei gradienti."""
        return self.net(x)
