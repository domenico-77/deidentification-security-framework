import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack


class SPSAAttack(BaseBlackBoxAttack):
    """
    SPSA (Simultaneous Perturbation Stochastic Approximation).

    Stima il gradiente della loss biometrica con 2 query per iterazione,
    e aggiorna x_adv con discesa del gradiente (segno).
    Usa la regione target-aware (`_get_perturbation_mask`) → ring per CIAGAN.
    """

    def __init__(self, detector_wrapper, anonymizer_fn, extract_emb_fn, compute_sim_fn,
                 device="cuda", n_iters=200, lr=2.0, delta=4.0,
                 region_mode="face", loss_mode="leakage"):
        super().__init__(
            detector_wrapper=detector_wrapper, device=device,
            n_queries=n_iters * 2,
            loss_mode=loss_mode, region_mode=region_mode,
            anonymizer_fn=anonymizer_fn,
            extract_emb_fn=extract_emb_fn,
            compute_sim_fn=compute_sim_fn,
        )
        self.n_iters = n_iters
        self.lr = lr
        self.delta = delta

    def _evaluate_loss(self, img_adv_tensor, emb_orig):
        """Wrapper di `_get_biometric_loss`: restituisce float (non tensore)."""
        loss = self._get_biometric_loss(img_adv_tensor, emb_orig)
        return float(loss)

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_adv = x_orig.clone()

        self._reset_embedding_cache()
        emb_orig = self._embed_orig(x_orig)
        if emb_orig is None:
            return x_orig, False, 0

        _, c, h, w = x_adv.shape

        # FIX: maschera target-aware (ring per CIAGAN, face per DP2)
        mask_base = self._get_perturbation_mask(x_orig)  # (1, H, W)

        # Adatta mask alla forma (1, C, H, W)
        mask = mask_base.unsqueeze(1).repeat(1, c, 1, 1)

        queries = 0
        for it in range(self.n_iters):
            # Rademacher ±1 * mask (perturba solo la regione target-aware)
            delta_vec = torch.randint(0, 2, (1, c, h, w),
                                     device=self.device).float() * 2.0 - 1.0
            delta_vec = delta_vec * mask

            # +c
            x_plus = torch.clamp(x_adv + self.delta * delta_vec, 0.0, 255.0)
            x_plus = torch.clamp(x_plus, x_orig - epsilon, x_orig + epsilon)
            loss_plus = self._evaluate_loss(x_plus, emb_orig)
            queries += 1

            # -c
            x_minus = torch.clamp(x_adv - self.delta * delta_vec, 0.0, 255.0)
            x_minus = torch.clamp(x_minus, x_orig - epsilon, x_orig + epsilon)
            loss_minus = self._evaluate_loss(x_minus, emb_orig)
            queries += 1

            # Stima gradiente SPSA
            grad_est = (loss_plus - loss_minus) / (2.0 * self.delta) * delta_vec

            # Aggiornamento (segno, non modulo, per rimanere in L∞)
            x_adv = x_adv - self.lr * torch.sign(grad_est)
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 255.0)

        # Success: soglia biometrica
        with torch.no_grad():
            final_loss = self._evaluate_loss(x_adv, emb_orig)
            final_sim = -final_loss
        success = final_sim > 0.55

        return x_adv, success, queries
