import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack


class SPSAAttack(BaseBlackBoxAttack):
    """
    Attacco SPSA (Simultaneous Perturbation Stochastic Approximation).
    Ottimizza una perturbazione per massimizzare la similarità d'identità in uscita
    dalla pipeline di anonimizzazione, sfruttando stime stocastiche del gradiente
    basate unicamente su valutazioni della loss.
    """

    def __init__(self, detector_wrapper, anonymizer_fn, extract_emb_fn, compute_sim_fn,
                 device="cuda", n_iters=50, lr=2.0, delta=1e-2):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=n_iters * 2)
        self.anonymizer_fn = anonymizer_fn
        self.extract_emb_fn = extract_emb_fn
        self.compute_sim_fn = compute_sim_fn
        self.n_iters = n_iters
        self.lr = lr
        self.delta = delta

    @staticmethod
    def _extract_image_tensor(output):
        """Estrae il tensore immagine da output che può essere:
        - torch.Tensor (DP2)
        - dict {"im": tensor} (CIAGAN)
        - tuple/list (fallback)
        """
        if isinstance(output, dict):
            for k in ("im", "image", "img", "output", "result", "images", "faces", "anonymized"):
                if k in output:
                    return output[k]
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
            raise ValueError(f"dict senza tensore: chiavi={list(output.keys())}")
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _evaluate_loss(self, img_adv_tensor, emb_orig):
        """
        L = -similarity(emb_orig, emb_anonymized_adversarial)
        Vogliamo minimizzare la loss (massimizzare la similarità).
        """
        with torch.no_grad():
            # 1. Prepara input uint8 per l'anonymizer
            img_single = img_adv_tensor[0] if img_adv_tensor.ndim == 4 else img_adv_tensor
            if img_single.max() <= 1.0:
                img_single = img_single * 255.0
            img_uint8 = img_single.detach().byte()

            # 2. Chiama la pipeline di anonimizzazione
            img_anonymized = self.anonymizer_fn(img_uint8)

            # 3. FIX: unwrap dict/tuple (CIAGAN restituisce {"im": tensor})
            img_anonymized = self._extract_image_tensor(img_anonymized)

            # 4. Normalizza a tensore CHW
            if isinstance(img_anonymized, torch.Tensor):
                img_anon_chw = img_anonymized.detach().cpu()
                if img_anon_chw.dim() == 4:
                    img_anon_chw = img_anon_chw.squeeze(0)
                if img_anon_chw.dtype == torch.uint8:
                    img_anon_chw = img_anon_chw.float()
            else:
                img_anon_chw = torch.from_numpy(np.array(img_anonymized)).permute(2, 0, 1).float()

            # 5. Embedding e similarità
            emb_anon = self.extract_emb_fn(img_anon_chw)

            if emb_orig is None or emb_anon is None:
                return 0.0

            sim = self.compute_sim_fn(emb_orig, emb_anon)
            return -float(sim)

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_adv = x_orig.clone()

        # Embedding di riferimento dal volto pulito
        img_clean_single = x_orig[0] if x_orig.ndim == 4 else x_orig
        if img_clean_single.max() <= 1.0:
            img_clean_single = img_clean_single * 255.0
        emb_orig = self.extract_emb_fn(img_clean_single.detach().cpu())

        if emb_orig is None:
            return x_orig, False, 0

        queries = 0
        _, c, h, w = x_adv.shape

        # ROI del volto per focalizzare la perturbazione
        ymin, xmin, ymax, xmax = self._get_face_roi(x_orig)

        for it in range(self.n_iters):
            # Rademacher ±1 (vettore di perturbazione ipo)
            delta_vec = torch.randint(0, 2, (1, c, h, w), device=self.device).float() * 2.0 - 1.0

            # Maschera: perturba solo la ROI del volto
            mask = torch.zeros((1, c, h, w), device=self.device)
            mask[:, :, ymin:ymax, xmin:xmax] = 1.0
            delta_vec = delta_vec * mask

            # Valutazione +c
            x_plus = torch.clamp(x_adv + self.delta * delta_vec, 0.0, 255.0)
            x_plus = torch.clamp(x_plus, x_orig - epsilon, x_orig + epsilon)
            loss_plus = self._evaluate_loss(x_plus, emb_orig)
            queries += 1

            # Valutazione -c
            x_minus = torch.clamp(x_adv - self.delta * delta_vec, 0.0, 255.0)
            x_minus = torch.clamp(x_minus, x_orig - epsilon, x_orig + epsilon)
            loss_minus = self._evaluate_loss(x_minus, emb_orig)
            queries += 1

            # Stima gradiente SPSA
            grad_est = (loss_plus - loss_minus) / (2.0 * self.delta) * delta_vec

            # Discesa del gradiente (per minimizzare loss = massimizzare similarità)
            x_adv = x_adv - self.lr * torch.sign(grad_est)
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 255.0)

        # Verifica evasione finale
        success = self._is_evaded(x_adv)
        return x_adv, success, queries
