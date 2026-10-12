import torch
import numpy as np
from abc import ABC, abstractmethod

class BaseBlackBoxAttack(ABC):
    """
    Classe base astratta per tutti gli attacchi black-box.
    Centralizza i metodi di utilità per l'interazione con il detector e il tracciamento delle query.
    """
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.n_queries = n_queries

    def _parse_detections(self, dets):
        """
        Estrae i box di rilevamento gestendo direttamente l'oggetto FaceDetection o tensori.
        """
        if dets is None:
            return []
        
        # Se l'oggetto è un'istanza di FaceDetection o ha attributi interni con i box
        for attr in ["boxes", "detections", "data", "dets", "_boxes", "cpu"]:
            if hasattr(dets, attr):
                val = getattr(dets, attr)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        continue
                if val is not None and not callable(val):
                    if isinstance(val, (np.ndarray, torch.Tensor, list, tuple)) and len(val) > 0:
                        dets = val
                        break

        if isinstance(dets, torch.Tensor):
            dets = dets.detach().cpu().numpy()
        
        if hasattr(dets, "cpu") and callable(dets.cpu):
            dets = dets.cpu().numpy()

        if isinstance(dets, (list, tuple, np.ndarray)):
            boxes = []
            for item in dets:
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                if hasattr(item, "__len__") and not isinstance(item, (str, bytes)):
                    for box in item:
                        if hasattr(box, "__len__") and len(box) >= 5:
                            boxes.append(box)
                elif hasattr(item, "__len__") and len(item) >= 5:
                    boxes.append(item)
            return boxes
            
        return []

    def _get_loss(self, img_tensor):
        """
        Calcola una loss proxy score-based basata sulla somma delle confidenze dei box rilevati.
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            
            if len(boxes) == 0:
                return 0.0  # Faccia evitata con successo (loss minima)
            
            score_sum = 0.0
            for box in boxes:
                if len(box) >= 5:
                    score_sum += float(box[4])
                else:
                    score_sum += 1.0
            return score_sum

    def _is_evaded(self, img_tensor):
        """
        Verifica binaria (decision-based) se il detector è stato eluso con successo.
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            dets = self.detector_wrapper(eval_input)
            boxes = self._parse_detections(dets)
            return len(boxes) == 0

    @abstractmethod
    def perturb(self, img_tensor, epsilon=16.0):
        """
        Metodo astratto che deve essere implementato da ogni attacco specifico.
        Deve restituire: (img_adv, success_bool, queries_count)
        """
        pass
