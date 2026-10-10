import torch
import torch.nn as nn
import sys
from pathlib import Path
from detectors.base_detector import BaseDetector

# Aggiungiamo la root del progetto al path (se non già presente) per garantire che 'models' sia visibile
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from models.retinaface import RetinaFace
except ImportError as e:
    print(f"[ERROR] Impossibile importare RetinaFace da models.retinaface: {e}")
    raise e

class RetinaFaceDetector(BaseDetector):
    """
    Wrapper per RetinaFace (MobileNet0.25) ottimizzato per il calcolo dei gradienti 
    tramite estrazione di feature intermedie o output diretti (feature/output-level attack).
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        
        try:
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
            # Inizializziamo il modello in fase di test
            self.net = RetinaFace(cfg=cfg, phase='test')
            
            # Caricamento dei pesi
            checkpoint = torch.load(weights_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            self.net.load_state_dict(new_state_dict, strict=False)
            
            self.net = self.net.to(self.device).eval()
            
            # Variabili per la gestione delle feature o loss
            self.clean_features = None
            self.extracted_features = None
            self._register_hook()
            
        except Exception as e:
            print(f"[ERROR] Impossibile caricare il modello RetinaFace da {weights_path}: {e}")
            raise e

    def _register_hook(self):
        """
        Registra un forward hook su un blocco intermedio della backbone o FPN
        per catturare i tensori intermedi ed evitare problemi di gradiente sulla head.
        """
        if hasattr(self.net, 'body') and hasattr(self.net.body, 'stage3'):
            target_layer = self.net.body.stage3
            
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
        """Restituisce il numero di volti rilevati tramite predizione RetinaFace standard."""
        with torch.no_grad():
            if image_tensor.dim() == 5:
                b, n, c, h, w = image_tensor.shape
                image_tensor = image_tensor.view(b * n, c, h, w)

            out = self.net(image_tensor)
            if isinstance(out, tuple) and len(out) >= 2:
                conf = out[1]
                if conf.dim() == 3:
                    valid = (conf[:, :, 1] > 0.5).sum().item()
                    return int(valid)
        return 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calcola la loss di attacco massimizzando la divergenza delle feature intermedie
        o sfruttando l'output grezzo della rete.
        """
        if image_tensor.dim() == 5:
            b, n, c, h, w = image_tensor.shape
            image_tensor = image_tensor.view(b * n, c, h, w)
            
        out = self.net(image_tensor)
        
        if hasattr(self, 'extracted_features') and self.extracted_features is not None:
            if not hasattr(self, 'clean_features') or self.clean_features is None:
                self.clean_features = self.extracted_features.detach()
            
            loss = -torch.nn.functional.mse_loss(self.extracted_features, self.clean_features)
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if isinstance(out, tuple):
                for t in out:
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        loss = loss + torch.relu(t).sum()
            elif isinstance(out, torch.Tensor):
                loss = torch.relu(out).sum()
                
        return loss

    def __call__(self, x):
        if x.dim() == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        return self.net(x)
