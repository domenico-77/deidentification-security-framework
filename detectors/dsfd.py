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
        """Estrae in modo sicuro la vera nn.Module di PyTorch bypassando i wrapper di DeepPrivacy2."""
        # 1. Cerca il detector specifico nel dizionario di DeepPrivacy2
        detector_obj = None
        if hasattr(wrapper, "detectors") and FaceDetection in wrapper.detectors:
            detector_obj = wrapper.detectors[FaceDetection]
        elif hasattr(wrapper, "face_detector"):
            detector_obj = wrapper.face_detector
        elif hasattr(wrapper, "detector"):
            detector_obj = wrapper.detector
        else:
            detector_obj = wrapper

        # 2. Se l'oggetto ha un attributo .net (tipico dei detector in DP2/face-detection), prendilo
        if hasattr(detector_obj, "net") and isinstance(detector_obj.net, nn.Module):
            return detector_obj.net

        # 3. Se ha un attributo .model ed è un nn.Module
        if hasattr(detector_obj, "model") and isinstance(detector_obj.model, nn.Module):
            return detector_obj.model

        # 4. Navigazione ricorsiva standard se è già un nn.Module
        if isinstance(detector_obj, nn.Module):
            return detector_obj

        # Se fallisce tutto, restituisce l'oggetto così com'è
        return detector_obj

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        # Usa il metodo di alto livello nativo del wrapper se disponibile per un conteggio sicuro delle box
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

        # Fallback manuale tramite forward sulla rete pura
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_bgr = img[:, [2, 1, 0], :, :]
        if img_bgr.max() <= 1.0:
            img_bgr = img_bgr * 255.0
            
        img_norm = img_bgr - self.mean.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_norm)

        count = 0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple) and len(out) > 0:
                    conf = out[0]
                    probs = torch.sigmoid(conf)
                    count += (probs > 0.5).sum().item()

        return max(1, count // 10) if count > 0 else 0

    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        img = image_tensor.to(self.device)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Normalizzazione DSFD (BGR, scala 0-255 e sottrazione media)
        img_bgr = img[:, [2, 1, 0], :, :]
        if img_bgr.max() <= 1.0:
            img_bgr = img_bgr * 255.0
            
        img_norm = img_bgr - self.mean.to(self.device)

        outputs = self.model(img_norm)

        # Loss: Soppressione delle logit di confidenza delle bounding box
        loss = 0.0
        if isinstance(outputs, (list, tuple)):
            for out in outputs:
                if isinstance(out, tuple) and len(out) > 0:
                    conf = out[0]
                    loss += torch.logsumexp(conf, dim=-1).mean()
        return loss
