import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector


class RetinaFaceDetector(BaseDetector):
    """
    Wrapper per RetinaFace (MobileNet0.25) da utilizzare come modello surrogato 
    per i Transfer Attack, implementando l'interfaccia BaseDetector.

    Espone anche un'interfaccia `nn.Module`-like (to, eval, train, zero_grad,
    parameters, state_dict) perché gli attacchi in `attacks/transfer/*` la
    assumono implicitamente (`surrogate_net.to(device)`, `surrogate_net.zero_grad()`).
    """

    def __init__(self, weights_path: str, device):
        # FIX: accetta sia str ("cuda"/"cpu") che torch.device
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.net = self._load_retinaface(weights_path)

    def _load_retinaface(self, weights_path: str) -> nn.Module:
        """Inizializza l'architettura e carica i pesi."""
        try:
            from models.retinaface import RetinaFace
        except ModuleNotFoundError as e:
            if "models" in str(e):
                raise RuntimeError(
                    "Impossibile importare `models.retinaface`. "
                    "Assicurati che la cartella del repo RetinaFace sia nel sys.path "
                    "e che contenga `models/retinaface.py`. "
                    f"Dettaglio: {e}"
                ) from e
            raise
        except ImportError as e:
            raise RuntimeError(
                "Errore di import interno a `models/retinaface.py`. "
                "Spesso è causato da `layers.*` o `utils.*` non risolvibili. "
                "Verifica che la root del repo RetinaFace sia nel sys.path. "
                f"Dettaglio: {e}"
            ) from e

        cfg = {
            'name': 'mobilenet0.25',
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True,
            'batch_size': 32,
            'ngpu': 1,
            'epoch': 250,
            'decay1': 190,
            'decay2': 220,
            'image_size': 640,
            'pretrain': False,
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64,
        }
        net = RetinaFace(cfg=cfg, phase='test')

        # FIX: map_location sicuro (self.device ora è un torch.device)
        map_location = 'cpu' if (self.device.type == 'cuda' and not torch.cuda.is_available()) else self.device
        checkpoint = torch.load(weights_path, map_location=map_location)

        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        net.load_state_dict(new_state_dict, strict=False)

        return net.to(self.device).eval()

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        """Restituisce il numero di volti rilevati (conf > 0.5)."""
        with torch.no_grad():
            out = self.net(image_tensor)
            if isinstance(out, tuple) and len(out) >= 2:
                conf = out[1]
                if conf.dim() == 3:
                    valid = (conf[:, :, 1] > 0.5).sum().item()
                    return int(valid)
        return 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Loss da massimizzare per far evadere il detector."""
        out = self.net(image_tensor)

        if isinstance(out, tuple):
            tensors = [t for t in out if isinstance(t, torch.Tensor)]
        elif isinstance(out, torch.Tensor):
            tensors = [out]
        else:
            return torch.zeros((), device=self.device, requires_grad=True)

        if not tensors:
            return torch.zeros((), device=self.device, requires_grad=True)

        loss = torch.relu(tensors[0]).sum()
        for t in tensors[1:]:
            loss = loss + torch.relu(t).sum()
        return loss

    # ---- nn.Module-like interface ----

    def to(self, *args, **kwargs):
        """No-op: il modello è già sul device corretto (impostato in __init__)."""
        return self

    def eval(self):
        self.net.eval()
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def zero_grad(self, *args, **kwargs):
        self.net.zero_grad(*args, **kwargs)

    def parameters(self):
        return self.net.parameters()

    def named_parameters(self):
        return self.net.named_parameters()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.net.load_state_dict(*args, **kwargs)

    def __call__(self, x):
        return self.net(x)
