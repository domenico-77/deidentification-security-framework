import torch
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack

class BoundaryAttack(BaseBlackBoxAttack):
    """
    Boundary Attack (Decision-based / Hard-Label Black-Box Attack).
    """
    def __init__(self, detector_wrapper, device="cuda", max_iter=200):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=max_iter)
        self.max_iter = max_iter

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        queries = 0

        x_adv = torch.clamp(x_orig + torch.randn_like(x_orig) * epsilon, 0.0, 255.0)
        x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
        
        queries += 1
        if not self._is_evaded(x_adv):
            x_adv = torch.clamp(x_orig + torch.sign(torch.randn_like(x_orig)) * epsilon, 0.0, 255.0)
            queries += 1
            if not self._is_evaded(x_adv):
                return x_orig, False, queries

        for it in range(1, self.max_iter + 1):
            perturbation = torch.randn_like(x_orig) * (epsilon * 0.1)
            x_try = torch.clamp(x_adv + perturbation, 0.0, 255.0)
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            
            queries += 1
            if self._is_evaded(x_try):
                x_adv = x_try
                
                alpha = 0.05
                x_projected = (1.0 - alpha) * x_adv + alpha * x_orig
                queries += 1
                if self._is_evaded(x_projected):
                    x_adv = x_projected

        success = self._is_evaded(x_adv)
        return x_adv, success, queries
