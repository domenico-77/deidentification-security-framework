import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector
from dp2.anonymizer.anonymizer import FaceDetection


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Estrae la rete interna (es. .net) se esiste, altrimenti usa il detector object
        self.model = self._extract_pure_nn_module(self.wrapper)
        
        # Assicura che self.model abbia sempre un metodo .eval() sicuro
        if not hasattr(self.model, "eval"):
            setattr(self.model, "eval", lambda: None)
        
        if hasattr(self.model, "to") and isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def _extract_pure_nn_module(self, wrapper):
        """Estrae la vera rete PyTorch interna (nn.Module) navigando gli attributi tipici di DSFD."""
        detector_obj = None
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
        elif hasattr(wrapper, "face_detector"):
            detector_obj = wrapper.face_detector
        elif hasattr(wrapper, "detector"):
            detector_obj = wrapper.detector
        else:
            detector_obj = wrapper

        # Cerca esplicitamente l'attributo .net che contiene il modello PyTorch di DSFD
        if hasattr(detector_obj, "net") and isinstance(detector_obj.net, nn.Module):
            return detector_obj.net
        if hasattr(detector_obj, "model") and isinstance(detector_obj.model, nn.Module):
            return detector_obj.model

        if isinstance(detector_obj, nn.Module):
            return detector_obj

        return detector_obj

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        detector_obj = None
        if hasattr(self.wrapper, "detectors") and FaceDetection in self.wrapper.detectors:
            detector_obj = self.wrapper.detectors[FaceDetection]
        elif hasattr(self.wrapper, "face_detector"):
            detector_obj = self.wrapper.face_detector

        if detector_obj is not None and hasattr(detector_obj, "detect_faces"):
            try:
                boxes, scores = detector_obj.detect_faces(image_tensor)
                return len(boxes) if boxes is not None else 0
            except Exception:
                pass

        return 1

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_bgr = img[:, [2, 1, 0], :, :]
        if img_bgr.max() <= 1.0:
            img_bgr = img_bgr * 255.0
            
        img_norm = img_bgr - self.mean.to(self.device)

        try:
            outputs = self.model(img_norm)
        except Exception:
            outputs = self.model(img)

        loss = 0.0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple) and len(out) > 0:
                    conf = out[0]
                    loss += torch.logsumexp(conf, dim=-1).mean()
        elif isinstance(outputs, torch.Tensor):
            loss += torch.logsumexp(outputs, dim=-1).mean()
            
        return loss
