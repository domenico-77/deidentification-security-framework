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
        Estrae i box di rilevamento gestendo in modo sicuro l'oggetto FaceDetection.
        """
        if dets is None:
            return []
        
        # Se l'oggetto ha un metodo o proprietà per convertirsi in numpy o tensor
        for meth in ["cpu", "numpy", "detach", "tolist"]:
            if hasattr(dets, meth) and callable(getattr(dets, meth)):
                try:
                    dets = getattr(dets, meth)()
                except Exception:
                    continue

        # Se l'oggetto è iterabile direttamente
        try:
            iterator = iter(dets)
        except TypeError:
            # Non iterabile: proviamo a cercare attributi interni o a convertirlo via dir()
            boxes = []
            for attr in dir(dets):
                if not attr.startswith("_"):
                    try:
                        val = getattr(dets, attr)
                        if isinstance(val, (np.ndarray, list, tuple)) and len(val) > 0:
                            for item in val:
                                if hasattr(item, "__len__") and len(item) >= 5:
                                    boxes.append(item)
                    except Exception:
                        continue
            return boxes

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
