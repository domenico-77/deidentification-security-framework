import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector
from dp2.anonymizer.anonymizer import FaceDetection


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._extract_detector_model(self.wrapper)
        
        # Sposta subito il modello sul device corretto e impostalo in eval
        if hasattr(self.model, "to"):
            self.model.to(self.device)
            self.model.eval()

        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def _extract_detector_model(self, wrapper):
        """Estrae in modo sicuro la rete neurale sottostante del detector DSFD dalla pipeline/wrapper."""
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
            if hasattr(detector_obj, "net"):
                return detector_obj.net
            if hasattr(detector_obj, "model"):
                return detector_obj.model
            return detector_obj

        if hasattr(wrapper, "face_detector"):
            fd = wrapper.face_detector
            return getattr(fd, "net", getattr(fd, "model", fd))
            
        if hasattr(wrapper, "detector"):
            det = wrapper.detector
            return getattr(det, "net", getattr(det, "model", det))

        if hasattr(wrapper, "net"):
            return wrapper.net

        return wrapper

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        # Assicura che il tensore sia sul device corretto
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Normalizzazione DSFD (BGR e sottrazione media)
        img_bgr = img[:, [2, 1, 0], :, :]
        img_norm = img_bgr - self.mean.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_norm)

        count = 0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple):  # [conf, loc]
                    conf = out[0]
                    probs = torch.sigmoid(conf)
                    count += (probs > 0.5).sum().item()
        
        if count == 0 and hasattr(self.wrapper, "detectors") and FaceDetection in self.wrapper.detectors:
            try:
                detector_obj = self.wrapper.detectors[FaceDetection]
                if hasattr(detector_obj, "detect"):
                    boxes = detector_obj.detect(image_tensor)
                    return len(boxes) if boxes is not None else 0
            except Exception:
                pass

        return max(1, count // 10) if count > 0 else 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # Assicura che il tensore sia sul device corretto
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Normalizzazione DSFD (BGR e sottrazione media)
        img_bgr = img[:, [2, 1, 0], :, :]
        img_norm = img_bgr - self.mean.to(self.device)

        outputs = self.model(img_norm)

        # Loss: Soppressione delle logit di confidenza delle bounding box
        loss = 0.0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple):  # [conf, loc]
                    conf = out[0]
                    loss += torch.logsumexp(conf, dim=-1).mean()
        return loss
