import torch
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

    def _get_loss(self, img_tensor):
        """
        Calcola una loss proxy score-based basata sulla somma delle confidenze dei box rilevati.
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                return 0.0  # Faccia evitata con successo (loss minima)
            
            score_sum = 0.0
            for det in dets:
                if len(det) > 0:
                    for box in det:
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
            return dets is None or len(dets) == 0 or len(dets[0]) == 0

    @abstractmethod
    def perturb(self, img_tensor, epsilon=16.0):
        """
        Metodo astratto che deve essere implementato da ogni attacco specifico.
        Deve restituire: (img_adv, success_bool, queries_count)
        """
        pass
