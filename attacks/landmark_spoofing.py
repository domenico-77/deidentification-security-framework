%%writefile attacks/landmark_spoofing.py
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

    def perturb(self, img_orig_tensor, epsilon=16.0, num_iter=20):
        """
        Attacco di landmark spoofing basato su perturbazione localizzata e 
        spostamento indotto dei punti di repere di dlib.
        """
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = num_iter

        # Otteniamo i landmark iniziali sull'immagine pulita per identificare la zona del viso
        img_np = img_orig_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        rects = self.dlib_detector(gray, 1)

        if not rects:
            return img_adv, True, 1  # Già evaso in partenza

        det = rects[0]
        shape = self.predictor(gray, det)
        
        # Estraiamo il bounding box della zona degli occhi e del naso (landmark 17 a 47)
        pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(17, 48)])
        x_min, y_min = np.clip(pts[:, 0].min() - 5, 0, img_np.shape[1]), np.clip(pts[:, 1].min() - 5, 0, img_np.shape[0])
        x_max, y_max = np.clip(pts[:, 0].max() + 5, 0, img_np.shape[1]), np.clip(pts[:, 1].max() + 5, 0, img_np.shape[0])

        # Ottimizzazione iterativa black-box / random search guidata sulla regione d'interesse
        best_img = img_adv.clone()
        
        for i in range(num_iter):
            # Generiamo una perturbazione mirata attorno alla zona oculare/nasale
            noise = torch.randn_like(img_adv) * (epsilon / 4.0)
            
            # Applichiamo la patch o il rumore prevalentemente sulla zona dei landmark
            patch_mask = torch.zeros_like(img_adv)
            patch_mask[:, y_min:y_max, x_min:x_max] = 1.0
            
            candidate = torch.clamp(img_adv + noise * patch_mask, 0.0, 255.0)
            
            # Verifica se la perturbazione ha fatto fallire il detector o alterato i landmark
            cand_np = candidate.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            cand_gray = cv2.cvtColor(cand_np, cv2.COLOR_RGB2GRAY)
            cand_rects = self.dlib_detector(cand_gray, 1)

            if not cand_rects:
                success = True
                success_iteration = i + 1
                img_adv = candidate
                break
            
            # Se il volto viene ancora rilevato, verifichiamo lo shift dei landmark
            new_shape = self.predictor(cand_gray, cand_rects[0])
            shift = sum(abs(shape.part(k).x - new_shape.part(k).x) + abs(shape.part(k).y - new_shape.part(k).y) for k in range(68))
            
            if shift > 35:  # Soglia di distorsione geometrica significativa
                success = True
                success_iteration = i + 1
                img_adv = candidate
                break

            img_adv = candidate

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=16.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
