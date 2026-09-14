import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector
from dp2.anonymizer.anonymizer import FaceDetection


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._extract_pure_nn_module(self.wrapper)
        
        if not hasattr(self.model, "eval"):
            setattr(self.model, "eval", lambda: None)
        
        if hasattr(self.model, "to") and isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def _extract_pure_nn_module(self, wrapper):
        """Esplora la struttura di DeepPrivacy2 per estrarre la vera nn.Module di DSFD."""
        detector_obj = None
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
        elif hasattr(wrapper, "face_detector"):
            detector_obj = wrapper.face_detector
        elif hasattr(wrapper, "detector"):
            detector_obj = wrapper.detector
        else:
            detector_obj = wrapper

        for attr in ["net", "model", "detector", "backbone", "body"]:
            if hasattr(detector_obj, attr):
                val = getattr(detector_obj, attr)
                if isinstance(val, nn.Module) and type(val).__name__ != 'FaceDetector':
                    return val

        queue = [detector_obj]
        visited = set()
        
        while queue:
            current = queue.pop(0)
            if id(current) in visited:
                continue
            visited.add(id(current))
            
            if isinstance(current, nn.Module) and type(current).__name__ != 'FaceDetector':
                return current
                
            for attr_name in dir(current):
                if attr_name.startswith('_'):
                    continue
                try:
                    val = getattr(current, attr_name)
                    if isinstance(val, nn.Module) and type(val).__name__ != 'FaceDetector':
                        return val
                    if hasattr(val, '__dict__') and id(val) not in visited:
                        queue.append(val)
                except Exception:
                    continue

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

        # Invocazione della forward pass passando le soglie richieste da SSD/DSFD
        try:
            outputs = self.model(img_norm, 0.01, 0.45)
        except TypeError:
            try:
                outputs = self.model(img_norm, confidence_threshold=0.01, nms_threshold=0.45)
            except Exception:
                outputs = self.model(img_norm)

        # Loss: Soppressione delle logit di confidenza delle bounding box
        loss = 0.0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple) and len(out) > 0:
                    conf = out[0]
                    loss += torch.logsumexp(conf, dim=-1).mean()
        elif isinstance(outputs, torch.Tensor):
            loss += torch.logsumexp(outputs, dim=-1).mean()
            
        return loss
