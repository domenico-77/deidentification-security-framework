import torch
import numpy as np
from attacks.blackbox.base_blackbox import BaseBlackBoxAttack

class SPSAAttack(BaseBlackBoxAttack):
    """
    Attacco SPSA (Simultaneous Perturbation Stochastic Approximation).
    Ottimizza una perturbazione per massimizzare la similarità d'identità in uscita da DeepPrivacy2,
    sfruttando stime stocastiche del gradiente basate unicamente su valutazioni della loss.
    """
    def __init__(self, detector_wrapper, anonymizer_fn, extract_emb_fn, compute_sim_fn, device="cuda", n_iters=50, lr=2.0, delta=1e-2):
        super().__init__(detector_wrapper=detector_wrapper, device=device, n_queries=n_iters * 2)
        self.anonymizer_fn = anonymizer_fn
        self.extract_emb_fn = extract_emb_fn
        self.compute_sim_fn = compute_sim_fn
        self.n_iters = n_iters
        self.lr = lr
        self.delta = delta

    def _evaluate_loss(self, img_adv_tensor, emb_orig):
        """
        Calcola la loss basata sull'output di DeepPrivacy2:
        L = - similarity(emb_orig, emb_anonymized_adversarial)
        Vogliamo minimizzare la loss (quindi massimizzare la similarità).
        """
        with torch.no_grad():
            # Assicura formato uint8 per l'anonymizer
            img_single = img_adv_tensor[0] if img_adv_tensor.ndim == 4 else img_adv_tensor
            if img_single.max() <= 1.0:
                img_single = img_single * 255.0
            img_uint8 = img_single.detach().byte()
            
            # Esegui la pipeline di DeepPrivacy2
            img_anonymized = self.anonymizer_fn(img_uint8)
            
            if isinstance(img_anonymized, torch.Tensor):
                img_anon_chw = img_anonymized.detach().cpu()
                if img_anon_chw.dim() == 4: 
                    img_anon_chw = img_anon_chw.squeeze(0)
            else:
                img_anon_chw = torch.from_numpy(np.array(img_anonymized)).permute(2, 0, 1)
            
            # Estrai embedding del volto restituito da DP2
            emb_anon = self.extract_emb_fn(img_anon_chw)
            
            if emb_orig is None or emb_anon is None:
                return 0.0
            
            sim = self.compute_sim_fn(emb_orig, emb_anon)
            # Loss negativa perché vogliamo massimizzare la similarità
            return -float(sim)

    def perturb(self, img_tensor, epsilon=16.0):
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_adv = x_orig.clone()
        
        # Estrai embedding di riferimento dell'immagine originale pulita
        img_clean_single = x_orig[0] if x_orig.ndim == 4 else x_orig
        if img_clean_single.max() <= 1.0:
            img_clean_single = img_clean_single * 255.0
        emb_orig = self.extract_emb_fn(img_clean_single.detach().cpu())
        
        if emb_orig is None:
            return x_orig, False, 0

        queries = 0
        _, c, h, w = x_adv.shape

        # Ottieni la ROI del volto per focalizzare la perturbazione
        ymin, xmin, ymax, xmax = self._get_face_roi(x_orig)

        for it in range(self.n_iters):
            # Genera vettore di perturbation stocastico (Rademacher: -1 o +1)
            delta_vec = torch.randint(0, 2, (1, c, h, w), device=self.device).float() * 2.0 - 1.0
            
            # Applica maschera di perturbaizione solo sulla ROI del volto
            mask = torch.zeros((1, c, h, w), device=self.device)
            mask[:, :, ymin:ymax, xmin:xmax] = 1.0
            delta_vec = delta_vec * mask

            # 1. Valutazione positiva (+c)
            x_plus = torch.clamp(x_adv + self.delta * delta_vec, 0.0, 255.0)
            x_plus = torch.clamp(x_plus, x_orig - epsilon, x_orig + epsilon)
            loss_plus = self._evaluate_loss(x_plus, emb_orig)
            queries += 1

            # 2. Valutazione negativa (-c)
            x_minus = torch.clamp(x_adv - self.delta * delta_vec, 0.0, 255.0)
            x_minus = torch.clamp(x_minus, x_orig - epsilon, x_orig + epsilon)
            loss_minus = self._evaluate_loss(x_minus, emb_orig)
            queries += 1

            # Stima del gradiente SPSA
            grad_est = (loss_plus - loss_minus) / (2.0 * self.delta) * delta_vec

            # Aggiornamento iterativo (Gradient Descent sulla loss negativa -> Massimizzazione similarità)
            x_adv = x_adv - self.lr * torch.sign(grad_est)
            x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 255.0)

        # Verifica finale se il detector è stato eluso o se l'identità è stata preservata
        success = self._is_evaded(x_adv)
        return x_adv, success, queries
