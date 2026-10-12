import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseBlackBoxAttack(ABC):
    """
    Classe base astratta per tutti gli attacchi black-box con supporto alla ROI del volto.
    """
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.n_queries = n_queries

    def _parse_detections(self, dets):
        if dets is None:
            return []
        
        if hasattr(dets, "numpy") and callable(dets.numpy):
            try:
                dets = dets.numpy()
            except Exception:
                pass

        if isinstance(dets, torch.Tensor):
            dets = dets.detach().cpu().numpy()

        if isinstance(dets, np.ndarray):
            if dets.size == 0:
                return []
            return dets.tolist()

        boxes = []
        for attr in ["boxes", "detections", "data", "dets", "_boxes", "pred_boxes"]:
            if hasattr(dets, attr):
                val = getattr(dets, attr)
                if val is not None:
                    if isinstance(val, torch.Tensor):
                        val = val.detach().cpu().numpy()
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    if isinstance(val, (list, tuple)):
                        boxes.extend(val)

        if len(boxes) > 0:
            return boxes

        if isinstance(dets, (list, tuple)):
            for item in dets:
                if hasattr(item, "__len__") and not isinstance(item, (str, bytes)):
                    boxes.append(item)
            return boxes

        return []

    def _get_face_roi(self, img_tensor):
        """
        Individua il bounding box del volto per focalizzare la perturbazione.
        Restituisce [ymin, xmin, ymax, xmax] interi (clampati alle dimensioni dell'immagine).
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            
            _, _, h, w = img_tensor.shape if img_tensor.ndim == 4 else (1, img_tensor.shape[0], img_tensor.shape[1], img_tensor.shape[2])
            
            if len(boxes) > 0 and len(boxes[0]) >= 4:
                b = boxes[0]
                ymin, xmin, ymax, xmax = int(b[1]), int(b[0]), int(b[3]), int(b[2]) # Adatta a seconda dell'ordine xyxy/yxyx
                # Fallback sicuro se invertiti
                if ymin > ymax: ymin, ymax = ymax, ymin
                if xmin > xmax: xmin, xmax = xmax, xmin
                
                # Margine di sicurezza del 20%
                pad_h = int((ymax - ymin) * 0.2)
                pad_w = int((xmax - xmin) * 0.2)
                
                ymin = max(0, ymin - pad_h)
                xmin = max(0, xmin - pad_w)
                ymax = min(h, ymax + pad_h)
                xmax = min(w, xmax + pad_w)
                
                return [ymin, xmin, ymax, xmax]
                
        # Fallback: intera immagine se nessun volto viene trovato al primo colpo
        return [0, 0, h, w]

    def _get_loss(self, img_tensor):
        """
        Loss proxy basata sul numero di box rilevati e sulla confidenza massima.
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            
            if len(boxes) == 0:
                return 0.0  # Evasione riuscita
            
            max_conf = 0.0
            for box in boxes:
                if len(box) >= 5 and float(box[4]) > max_conf:
                    max_conf = float(box[4])
            
            return float(len(boxes)) * 10.0 + max_conf

    def _is_evaded(self, img_tensor):
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            return len(boxes) == 0

    @abstractmethod
    def perturb(self, img_tensor, epsilon=16.0):
        pass
