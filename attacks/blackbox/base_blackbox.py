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
        Estrae i box di rilevamento gestendo robustamente l'oggetto FaceDetection.
        """
        if dets is None:
            return []
        
        # Se l'oggetto ha una proprietà o metodo numpy / tensor
        if hasattr(dets, "numpy") and callable(dets.numpy):
            try:
                dets = dets.numpy()
            except Exception:
                pass

        # Se è già un tensore PyTorch
        if isinstance(dets, torch.Tensor):
            dets = dets.detach().cpu().numpy()

        # Se è un numpy array
        if isinstance(dets, np.ndarray):
            if dets.size == 0:
                return []
            return dets.tolist()

        # Se è un oggetto FaceDetection, proviamo a estrarre i box dagli attributi interni standard
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

        # Tentativo finale: se l'oggetto stesso è una tupla o lista di box
        if isinstance(dets, (list, tuple)):
            for item in dets:
                if hasattr(item, "__len__") and not isinstance(item, (str, bytes)):
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
