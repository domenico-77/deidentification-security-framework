%%writefile attacks/blackbox/simba_attack.py
import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack


class SimBA(BaseBlackBoxAttack):
    """
    SimBA (Simultaneous Byte-wise Attack) con supporto patch.

    Se `patch_size=1` → SimBA classico (1 pixel per query).
    Se `patch_size>1` → SimBA-patch: aggiorna una patch N×N per query,
                        N×N volte più efficiente in copertura dello spazio.

    Supporta `loss_mode` ("evasion" o "leakage") e `region_mode` ("face" o "ring").
    """

    def __init__(self, detector_wrapper, device="cuda", n_queries=1000, step_size=2.0,
                 patch_size=4, loss_mode="evasion", region_mode="face",
                 anonymizer_fn=None, extract_emb_fn=None, compute_sim_fn=None):
        super().__init__(
            detector_wrapper=detector_wrapper, device=device, n_queries=n_queries,
            loss_mode=loss_mode, region_mode=region_mode,
            anonymizer_fn=anonymizer_fn, extract_emb_fn=extract_emb_fn,
            compute_sim_fn=compute_sim_fn,
        )
        self.step_size = step_size
        self.patch_size = max(1, int(patch_size))

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_best = x_orig.clone()

        self._reset_embedding_cache()

        emb_orig = None
        if self.loss_mode == "leakage":
            emb_orig = self._embed_orig(x_orig)
            if emb_orig is None:
                return x_orig, False, 0

        best_loss = self._get_loss(x_best, emb_orig=emb_orig)
        if self.loss_mode == "evasion" and best_loss == 0.0:
            return x_best, True, 0

        _, c, h, w = x_best.shape

        # Maschera target-aware (ring per CIAGAN, face per DP2)
        mask = self._get_perturbation_mask(x_orig)
        mask_np = mask[0].cpu().numpy()
        ys, xs = np.where(mask_np > 0.5)
        if len(ys) == 0 or len(xs) == 0:
            ymin, xmin, ymax, xmax = 0, 0, h, w
        else:
            ymin, xmin = int(ys.min()), int(xs.min())
            ymax, xmax = int(ys.max()) + 1, int(xs.max()) + 1

        # Clamp patch_size per non eccedere la regione
        p = self.patch_size
        max_p = min(ymax - ymin, xmax - xmin)
        if p > max_p:
            p = max(1, max_p)

        queries = 0
        for query_idx in range(1, self.n_queries + 1):
            queries = query_idx

            # Posizione casuale della patch dentro la regione perturbabile
            # (se patch_size=1, degenera in un singolo pixel — SimBA classico)
            h_idx = np.random.randint(ymin, max(ymin + 1, ymax - p + 1))
            w_idx = np.random.randint(xmin, max(xmin + 1, xmax - p + 1))

            # SimBA-patch: aggiorna TUTTI i canali della patch insieme
            direction = torch.zeros((1, c, h, w), device=self.device)
            direction[0, :, h_idx:h_idx + p, w_idx:w_idx + p] = 1.0

            # Prova -step
            x_try = x_best - self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_sub = self._get_loss(x_try, emb_orig=emb_orig)

            if loss_sub < best_loss:
                best_loss = loss_sub
                x_best = x_try
                if self.loss_mode == "evasion" and best_loss == 0.0:
                    return x_best, True, queries
                continue

            # Prova +step
            x_try = x_best + self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_add = self._get_loss(x_try, emb_orig=emb_orig)

            if loss_add < best_loss:
                best_loss = loss_add
                x_best = x_try
                if self.loss_mode == "evasion" and best_loss == 0.0:
                    return x_best, True, queries

        # Success coerente con loss_mode
        if self.loss_mode == "leakage":
            success = (-best_loss) > 0.55
        else:
            success = self._is_evaded(x_best)

        return x_best, success, queries
