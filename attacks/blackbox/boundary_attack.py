import torch
import numpy as np

class BoundaryAttack:
    """
    Boundary Attack (Decision-based / Hard-Label Black-Box Attack).
    Richiede solo l'output binario (rilevato / non rilevato) del detector,
    muovendosi lungo il confine decisionale per trovare la perturbazione minima.
    """
    def __init__(self, detector_wrapper, device="cuda", max_iter=200, step_adap=1.0):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.max_iter = max_iter
        self.step_adap = step_adap

    def _is_evaded(self, img_tensor):
        """Restituisce True se il detector non rileva alcuna faccia (attacco riuscito)."""
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            dets = self.detector_wrapper(eval_input)
            return dets is None or len(dets) == 0 or len(dets[0]) == 0

    def perturb(self, img_tensor, epsilon=16.0):
        """
        Esegue il Boundary Attack su una singola immagine.
        Restituisce: (img_adv, success, queries_count)
        """
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        queries = 0

        # 1. Trova un punto iniziale che già evade il detector (es. aggiungendo rumore casuale limitato da epsilon)
        x_adv = torch.clamp(x_orig + torch.randn_like(x_orig) * epsilon, 0.0, 255.0)
        x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
        
        queries += 1
        if not self._is_evaded(x_adv):
            # Se il rumore casuale non basta, proviamo a spingerlo al massimo limite di epsilon
            x_adv = torch.clamp(x_orig + torch.sign(torch.randn_like(x_orig)) * epsilon, 0.0, 255.0)
            queries += 1
            if not self._is_evaded(x_adv):
                # Se l'inizializzazione fallisce, restituiamo l'immagine originale o un fallimento controllato
                return x_orig, False, queries

        # 2. Ottimizzazione lungo il confine decisionale (Spherical interpolation & rejection sampling)
        for it in range(1, self.max_iter + 1):
            # Campionamento sferico / perturbazione gaussiana
            perturbation = torch.randn_file = torch.randn_like(x_orig) * (epsilon * 0.1)
            x_try = torch.clamp(x_adv + perturbation, 0.0, 255.0)
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            
            queries += 1
            if self._is_evaded(x_try):
                x_adv = x_try  # Rimaniamo nella regione avversaria e riduciamo la distanza dall'originale
                
                # Proiezione verso l'immagine originale per minimizzare la perturbazione
                alpha = 0.05
                x_projected = (1.0 - alpha) * x_adv + alpha * x_orig
                queries += 1
                if self._is_evaded(x_projected):
                    x_adv = x_projected

        success = self._is_evaded(x_adv)
        return x_adv, success, queries
