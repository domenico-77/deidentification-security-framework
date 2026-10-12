import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack

class SquareAttack(BaseBlackBoxAttack):
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000, p_init=0.8):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=n_queries)
        self.p_init = p_init

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
        ymin, xmin, ymax, xmax = self._get_face_roi(x_orig)
        roi_h, roi_w = ymax - ymin, xmax - xmin

        queries = 0
        for query_idx in range(1, self.n_queries + 1):
            queries = query_idx
            
            side_len = int(np.ceil(np.sqrt(self.p_init) * min(roi_h, roi_w)))
            side_len = max(1, min(side_len, min(roi_h, roi_w)))
            
            h_start = np.random.randint(ymin, max(ymin + 1, ymax - side_len + 1))
            w_start = np.random.randint(xmin, max(xmin + 1, xmax - side_len + 1))
            
            x_try = x_best.clone()
            perturbation = torch.empty((c, side_len, side_len), device=self.device).uniform_(-epsilon, epsilon)
            
            current_patch = x_try[:, :, h_start:h_start+side_len, w_start:w_start+side_len]
            new_patch = torch.clamp(current_patch + perturbation, 0.0, 255.0)
            x_try[:, :, h_start:h_start+side_len, w_start:w_start+side_len] = new_patch
            
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)
            
            current_loss = self._get_loss(x_try)
            
            if current_loss < best_loss:
                best_loss = current_loss
                x_best = x_try
                
                if best_loss == 0.0:
                    return x_best, True, queries

        success = self._is_evaded(x_best)
        return x_best, success, queries
