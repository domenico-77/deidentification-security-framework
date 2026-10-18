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

    def perturb(self, img_orig_tensor, epsilon=16.0, num_iter=30):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = num_iter

        # Estrazione dei landmark di riferimento sull'immagine originale
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

        for i in range(num_iter):
            step = (torch.rand_like(img_adv) * 2.0 - 1.0) * (epsilon / 3.0)
            
            mask = torch.zeros_like(img_adv)
            mask[:, y1:y2, x1:x2] = 1.0
            
            current_perturbation = current_perturbation + step * mask
            current_perturbation = torch.clamp(current_perturbation, min=-epsilon, max=epsilon)
            
            candidate = torch.clamp(img_orig_tensor + current_perturbation, min=0.0, max=255.0)

            cand_np = candidate.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            cand_gray = cv2.cvtColor(cand_np, cv2.COLOR_RGB2GRAY)
            cand_rects = self.dlib_detector(cand_gray, 1)

            # Se il detector perde il volto, è successo pieno
            if not cand_rects:
                success = True
                success_iteration = i + 1
                best_candidate = candidate
                break

            # Criterio di Landmark Spoofing: calcolo dello shift geometrico dei 68 punti
            try:
                new_shape = self.predictor(cand_gray, cand_rects[0])
                shift = sum(abs(orig_shape.part(k).x - new_shape.part(k).x) + 
                            abs(orig_shape.part(k).y - new_shape.part(k).y) for k in range(68))
                
                # Se i landmark si spostano significativamente (> 40 pixel cumulativi), la U-Net fallisce il conditioning
                if shift > 40:
                    success = True
                    success_iteration = i + 1
                    best_candidate = candidate
                    break
            except Exception:
                # Se la distorsione è talmente forte che il predittore fallisce
                success = True
                success_iteration = i + 1
                best_candidate = candidate
                break

            best_candidate = candidate

        return best_candidate, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=16.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
