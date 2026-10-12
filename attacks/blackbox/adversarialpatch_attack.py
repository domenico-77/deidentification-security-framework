import torch
import numpy as np
from attacks.blackbox.base import BaseBlackBoxAttack

class AdversarialPatchAttack(BaseBlackBoxAttack):
    def __init__(self, detector_wrapper, device="cuda"):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=1)

    def perturb(self, img_tensor, epsilon=16.0):
        """
        Applica una patch avversaria strutturata sulla regione centrale del volto.
        Ignora epsilon o lo usa per scalare il contrasto della patch.
        """
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_adv = img_tensor.clone()
        _, c, h, w = x_adv.shape
        
        # Ottieni la ROI del volto
        ymin, xmin, ymax, xmax = self._get_face_roi(x_adv)
        
        face_h = ymax - ymin
        face_w = xmax - xmin
        
        patch_h = max(10, int(face_h * 0.4))
        patch_w = max(10, int(face_w * 0.4))
        
        p_ymin = ymin + int((face_h - patch_h) / 2)
        p_xmin = xmin + int((face_w - patch_w) / 2)
        p_ymax = p_ymin + patch_h
        p_xmax = p_xmin + patch_w
        
        # Genera una patch avversaria (può usare epsilon per modulare l'intensità o valori pieni)
        if epsilon >= 200.0:
            patch = torch.randint(0, 256, (c, patch_h, patch_w), device=self.device).float()
        else:
            # Patch limitata dal budget di epsilon attorno all'immagine originale
            orig_patch = x_adv[:, :, p_ymin:p_ymax, p_xmin:p_xmax]
            noise = torch.empty((c, patch_h, patch_w), device=self.device).uniform_(-epsilon, epsilon)
            patch = torch.clamp(orig_patch + noise, 0.0, 255.0)
        
        x_adv[:, :, p_ymin:p_ymax, p_xmin:p_xmax] = patch
        
        success = self._is_evaded(x_adv)
        return x_adv, success, 1
