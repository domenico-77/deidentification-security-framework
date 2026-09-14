import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector
from dp2.anonymizer.anonymizer import FaceDetection


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._extract_detector_model(self.wrapper)
        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def _extract_detector_model(self, wrapper):
        """Estrae in modo sicuro la rete neurale sottostante del detector DSFD dalla pipeline/wrapper."""
        # 1. Se il wrapper è l'Anonymizer di DeepPrivacy2 con il dizionario detectors
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
            if hasattr(detector_obj, "net"):
                return detector_obj.net
            if hasattr(detector_obj, "model"):
                return detector_obj.model
            return detector_obj

        # 2. Se possiede direttamente l'attributo face_detector
        if hasattr(wrapper, "face_detector"):
            fd = wrapper.face_detector
            return getattr(fd, "net", getattr(fd, "model", fd))
            
        # 3. Fallback se passa un wrapper personalizzato o un oggetto diretto
        if hasattr(wrapper, "detector"):
            det = wrapper.detector
            return getattr(det, "net", getattr(det, "model", det))

        # 4. Fallback estremo se wrapper è già la rete o ha 'net'
        if hasattr(wrapper, "net"):
            return wrapper.net

        return wrapper

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        if hasattr(self.wrapper, "detect_faces"):
            boxes, scores = self.wrapper.detect_faces(image_tensor)
        else:
            # Se la pipeline non ha detect_faces diretto, prova a passare attraverso il detector object
            detector_obj = getattr(self.wrapper, "detectors", {}).get(FaceDetection, self.wrapper)
            if hasattr(detector_obj, "detect_faces"):
                boxes, scores = detector_obj.detect_faces(image_tensor)
            else:
                raise AttributeError("Impossibile trovare un metodo di rilevamento facciale valido nel wrapper.")
                
        return len(boxes) if boxes is not None else 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 3:
            img = image_tensor.unsqueeze(0)
        else:
            img = image_tensor

        # Normalizzazione DSFD (BGR e sottrazione media)
        img_bgr = img[:, [2, 1, 0], :, :]
        img_norm = img_bgr - self.mean

        outputs = self.model(img_norm)

        # Loss: Soppressione delle logit di confidenza delle bounding box
        loss = 0.0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple):  # [conf, loc]
                    conf = out[0]
                    loss += torch.logsumexp(conf, dim=-1).mean()
        return loss
