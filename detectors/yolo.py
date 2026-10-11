import torch
import torch.nn as nn
import subprocess
import sys
from detectors.base_detector import BaseDetector

try:
    from ultralytics import YOLO
except ImportError:
    print("[INFO] Libreria 'ultralytics' non trovata. Installazione automatica in corso...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

class YoloFaceDetector(BaseDetector):
    """
    Wrapper per YOLOv8-face ottimizzato per il calcolo dei gradienti 
    tramite estrazione di feature intermedie (feature-level attack).
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        
        try:
            self.yolo_instance = YOLO(weights_path)
            self.net = self.yolo_instance.model.to(self.device).eval()
            self.target_features = None
            self._register_hook()
            
        except Exception as e:
            print(f"[ERROR] Impossibile caricare il modello YOLO da {weights_path}: {e}")
            raise e

    def _register_hook(self):
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
        with torch.no_grad():
            if image_tensor.shape[-1] != 256 or image_tensor.shape[-2] != 256:
                image_tensor_resized = torch.nn.functional.interpolate(
                    image_tensor, size=(256, 256), mode='bilinear', align_corners=False
                )
            else:
                image_tensor_resized = image_tensor

            results = self.yolo_instance(image_tensor_resized, verbose=False)
            count = 0
            for r in results:
                if r.boxes is not None:
                    count += len(r.boxes)
            return count

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 5:
            b, n, c, h, w = image_tensor.shape
            image_tensor = image_tensor.view(b * n, c, h, w)
            
        _ = self.net(image_tensor)
        
        if not hasattr(self, 'clean_features') or self.clean_features is None:
            self.clean_features = self.extracted_features.detach()

        loss = -torch.nn.functional.mse_loss(self.extracted_features, self.clean_features)
        return loss

    def __call__(self, x):
        if x.dim() == 5:
            b, n, c, h, w = x.shape
            x = x.view(b * n, c, h, w)
        return self.net(x)
