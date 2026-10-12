import torch
from attacks.blackbox.base import BaseBlackBoxAttack

class SimBA(BaseBlackBoxAttack):
    """
    SimBA (Simple Black-Box Attack).
    """
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000, step_size=2.0):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=n_queries)
        self.step_size = step_size

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_best = x_orig.clone()
        
        best_loss = self._get_loss(x_best)
        if best_loss == 0.0:
            return x_best, True, 0

        _, c, h, w = x_best.shape
        queries = 0

        for query_idx in range(1, self.n_queries + 1):
            queries = query_idx

            c_idx = torch.randint(0, c, (1,)).item()
            h_idx = torch.randint(0, h, (1,)).item()
            w_idx = torch.randint(0, w, (1,)).item()

            direction = torch.zeros((1, c, h, w), device=self.device)
            direction[0, c_idx, h_idx, w_idx] = 1.0

            x_try = x_best - self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_sub = self._get_loss(x_try)

            if loss_sub < best_loss:
                best_loss = loss_sub
                x_best = x_try
                if best_loss == 0.0:
                    return x_best, True, queries
                continue

            x_try = x_best + self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_add = self._get_loss(x_try)

            if loss_add < best_loss:
                best_loss = loss_add
                x_best = x_try
                if best_loss == 0.0:
                    return x_best, True, queries

        success = self._is_evaded(x_best)
        return x_best, success, queries
