import torch
import numpy as np
import cv2
from attacks.base_attack import BaseAttack

class LandmarkSpoofingAttack(BaseAttack):
    def __init__(self, target=None, detector_wrapper=None, device: torch.device = None):
        det = detector_wrapper if detector_wrapper is not None else (target.detector_wrapper if target else None)
        super().__init__(detector=det, device=device)
        self.target = target
        self.dlib_detector = target.dlib_detector if target else None
        self.predictor = target.predictor if target else None

    def _compute_score(self, candidate, orig_shape, y1, y2, x1, x2):
        cand_np = candidate.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        cand_gray = cv2.cvtColor(cand_np, cv2.COLOR_RGB2GRAY)
        cand_rects = self.dlib_detector(cand_gray, 1)

        if not cand_rects:
            return 99999.0, cand_gray, None, True  # Evasione totale

        try:
            new_shape = self.predictor(cand_gray, cand_rects[0])
            shift_key = sum(abs(orig_shape.part(k).x - new_shape.part(k).x) + 
                            abs(orig_shape.part(k).y - new_shape.part(k).y) for k in range(17, 48))
            total_shift = sum(abs(orig_shape.part(k).x - new_shape.part(k).x) + 
                              abs(orig_shape.part(k).y - new_shape.part(k).y) for k in range(68))
            score = float(shift_key * 1.5 + total_shift)
            return score, cand_gray, new_shape, False
        except Exception:
            return 99999.0, cand_gray, None, True

    def perturb(self, img_orig_tensor, epsilon=16.0, num_iter=50):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = num_iter

        img_np = img_orig_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        rects = self.dlib_detector(gray, 1)

        if not rects:
            return img_adv, True, 1

        det = rects[0]
        orig_shape = self.predictor(gray, det)
        
        x1, y1, x2, y2 = max(0, det.left()), max(0, det.top()), min(img_np.shape[1], det.right()), min(img_np.shape[0], det.bottom())
        
        best_candidate = img_adv.clone()
        current_perturbation = torch.zeros_like(img_adv)
        best_score = 0.0

        for i in range(num_iter):
            # Direzione basata su segno per sfruttare efficacemente la epsilon-ball
            u = torch.sign(torch.randn_like(img_adv))
            mask = torch.zeros_like(img_adv)
            mask[:, y1:y2, x1:x2] = 1.0
            u = u * mask
            
            step_size = epsilon * 0.4
            
            # Test delle direzioni speculari plus/minus
            pert_plus = torch.clamp(current_perturbation + step_size * u, min=-epsilon, max=epsilon)
            pert_minus = torch.clamp(current_perturbation - step_size * u, min=-epsilon, max=epsilon)
            
            cand_plus = torch.clamp(img_orig_tensor + pert_plus, min=0.0, max=255.0)
            cand_minus = torch.clamp(img_orig_tensor + pert_minus, min=0.0, max=255.0)

            score_plus, _, _, evaded_plus = self._compute_score(cand_plus, orig_shape, y1, y2, x1, x2)
            score_minus, _, _, evaded_minus = self._compute_score(cand_minus, orig_shape, y1, y2, x1, x2)

            if score_plus > score_minus:
                current_perturbation = pert_plus
                best_candidate = cand_plus
                best_score = score_plus
            else:
                current_perturbation = pert_minus
                best_candidate = cand_minus
                best_score = score_minus

            if evaded_plus or evaded_minus or best_score > 130.0:
                success = True
                success_iteration = i + 1
                break

        return best_candidate, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=16.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
