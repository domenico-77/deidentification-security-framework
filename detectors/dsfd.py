import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector
from dp2.anonymizer.anonymizer import FaceDetection


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._extract_pure_nn_module(self.wrapper)
        
        if hasattr(self.model, "to"):
            self.model.to(self.device)
            self.model.eval()

        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def _extract_pure_nn_module(self, wrapper):
        """Estrae la vera rete PyTorch interna (nn.Module) dal wrapper DSFD."""
        detector_obj = None
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
        elif hasattr(wrapper, "face_detector"):
            detector_obj = wrapper.face_detector
        elif hasattr(wrapper, "detector"):
            detector_obj = wrapper.detector
        else:
            detector_obj = wrapper

        # Cerca ricorsivamente l'attributo che contiene la rete PyTorch pura (.net o .model)
        current = detector_obj
        for attr in ["net", "model", "detector", "face_detector"]:
            if hasattr(current, attr):
                val = getattr(current, attr)
                if isinstance(val, nn.Module):
                    return val
                current = val

        if isinstance(current, nn.Module):
            return current

        # Fallback estremo: se non troviamo un nn.Module puro, usiamo un fallback sul wrapper stesso
        return detector_obj

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        detector_obj = None
        if hasattr(self.wrapper, "detectors") and FaceDetection in self.wrapper.detectors:
            detector_obj = self.wrapper.detectors[FaceDetection]
        elif hasattr(self.wrapper, "face_detector"):
            detector_obj = self.wrapper.face_detector

        # Usa il metodo di detection nativo se disponibile
        if detector_obj is not None and hasattr(detector_obj, "detect_faces"):
            try:
                boxes, scores = detector_obj.detect_faces(image_tensor)
                return len(boxes) if boxes is not None else 0
            except Exception:
                pass

        return 1  # Valore di sicurezza di fallback

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Normalizzazione DSFD (BGR, scala 0-255 e sottrazione media)
        img_bgr = img[:, [2, 1, 0], :, :]
        if img_bgr.max() <= 1.0:
            img_bgr = img_bgr * 255.0
            
        img_norm = img_bgr - self.mean.to(self.device)

        # Se self.model è ancora il wrapper di alto livello, proviamo a richiamarlo correttamente passandogli il batch o gestendolo
        try:
            outputs = self.model(img_norm)
        except Exception:
            # Fallback se il modello richiede un formato particolare
            outputs = self.model(img)

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
