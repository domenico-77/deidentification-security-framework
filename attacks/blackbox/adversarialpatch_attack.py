import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack

class AdversarialPatchAttack(BaseBlackBoxAttack):
    def __init__(self, detector_wrapper, device="cuda"):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=1)

    def perturb(self, img_tensor, epsilon=255.0):
        """
        Applica una patch strutturata aggressiva (pattern a bande geometriche ad alto contrasto)
        sull'area centrale del volto per forzare l'elusione del detector DSFD.
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
        
        # Ampliamo la patch al 60% del volto per distruggere i punti di ancoraggio
        patch_h = max(20, int(face_h * 0.6))
        patch_w = max(20, int(face_w * 0.6))
        
        p_ymin = ymin + int((face_h - patch_h) / 2)
        p_xmin = xmin + int((face_w - patch_w) / 2)
        p_ymax = min(h, p_ymin + patch_h)
        p_xmax = min(w, p_xmin + patch_w)
        
        actual_h = p_ymax - p_ymin
        actual_w = p_xmax - p_xmin

        # Genera un pattern geometrico aggressivo ad alto contrasto (bande verticali bianche e nere alternate o saturazione estrema)
        patch = torch.zeros((c, actual_h, actual_w), device=self.device)
        for i in range(0, actual_w, 8):
            if (i // 8) % 2 == 0:
                patch[:, :, i:i+4] = 255.0
            else:
                patch[:, :, i:i+4] = 0.0

        # Sovrapponi la patch geometrica pesante sulla ROI
        x_adv[:, :, p_ymin:p_ymax, p_xmin:p_xmax] = patch
        
        # Verifica se il detector è stato finalmente eluso
        success = self._is_evaded(x_adv)
        
        return x_adv, success, 1
