import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack

class BoundaryAttack(BaseBlackBoxAttack):
    def __init__(self, detector_wrapper, device="cuda", max_iter=200):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=max_iter)
        self.max_iter = max_iter

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        _, _, h, w = x_orig.shape
        ymin, xmin, ymax, xmax = self._get_face_roi(x_orig)
        
        queries = 0
        x_adv = x_orig.clone()
        found_init = False

        # Inizializzazione robusta mirata sulla ROI del volto
        for _ in range(50):
            queries += 1
            noise = torch.randn_like(x_orig) * epsilon
            # Applica rumore principalmente sul volto
            mask = torch.zeros_like(x_orig)
            mask[:, :, ymin:ymax, xmin:xmax] = 1.0
            
            x_try = torch.clamp(x_orig + noise * mask, 0.0, 255.0)
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            
            if self._is_evaded(x_try):
                x_adv = x_try
                found_init = True
                break
        
        if not found_init:
            return x_orig, False, queries

        # Ottimizzazione iterativa sul confine
        for it in range(1, self.max_iter + 1):
            queries += 1
            perturbation = torch.randn_like(x_orig) * (epsilon * 0.1)
            x_try = torch.clamp(x_adv + perturbation, 0.0, 255.0)
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            
            if self._is_evaded(x_try):
                x_adv = x_try
                alpha = 0.05
                x_projected = (1.0 - alpha) * x_adv + alpha * x_orig
                queries += 1
                if self._is_evaded(x_projected):
                    x_adv = x_projected

        success = self._is_evaded(x_adv)
        return x_adv, success, queries
