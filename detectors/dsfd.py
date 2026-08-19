import torch
import torch.nn as nn
from detectors.base_detector import BaseDetector


class DSFDDetector(BaseDetector):
    def __init__(self, deeprivacy_wrapper, device=None):
        self.wrapper = deeprivacy_wrapper
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = getattr(self.wrapper.face_detector, 'net', self.wrapper.face_detector)
        self.mean = torch.tensor([104.0, 117.0, 123.0], device=self.device).view(1, 3, 1, 1)

    def count_detections(self, image_tensor: torch.Tensor) -> int:
        boxes, scores = self.wrapper.detect_faces(image_tensor)
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