import torch
import numpy as np

class SquareAttack:
    """
    Square Attack (Score-based Black-Box Attack).
    Genera perturbazioni avversarie modificando iterativamente regioni quadrate
    dell'immagine guidato esclusivamente dalle confidenze o dai rilevamenti del detector.
    """
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000, p_init=0.8):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.n_queries = n_queries
        self.p_init = p_init

    def _get_loss(self, img_tensor):
        """
        Calcola una loss proxy basata sul numero e sulla confidenza dei box rilevati.
        L'obiettivo è massimizzare la perdita di rilevamento (evasione).
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                return 0.0  # Faccia completamente evitata (loss minima per il detector, massima per l'attacco)
            
            # Somma delle confidenze dei box rilevati (da minimizzare per ingannare il detector)
            score_sum = 0.0
            for det in dets:
                if len(det) > 0:
                    for box in det:
                        if len(box) >= 5:
                            score_sum += float(box[4])
                        else:
                            score_sum += 1.0
            return score_sum

    def perturb(self, img_tensor, epsilon=16.0):
        """
        Esegue il Square Attack su una singola immagine.
        Restituisce: (img_adv, success, queries_count)
        """
        img_tensor = img_tensor.to(self.device).float()
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        x_orig = img_tensor.clone().detach()
        x_best = x_orig.clone()
        
        best_loss = self._get_loss(x_best)
        if best_loss == 0.0:
            return x_best, True, 0

        _, c, h, w = x_best.shape
        queries = 0

        for query_idx in range(1, self.n_queries + 1):
            queries = query_idx
            
            # Determina la dimensione del quadrato in base a p_init
            side_len = int(np.ceil(np.sqrt(self.p_init) * h))
            side_len = max(1, min(side_len, h))
            
            # Scegli coordinate casuali per il quadrato
            h_start = np.random.randint(0, h - side_len + 1)
            w_start = np.random.randint(0, w - side_len + 1)
            
            # Genera perturbazione casuale limitata da epsilon nel blocco
            x_try = x_best.clone()
            perturbation = torch.empty((c, side_len, side_len), device=self.device).uniform_(-epsilon, epsilon)
            
            # Applica il quadrato all'immagine candidata
            current_patch = x_try[:, :, h_start:h_start+side_len, w_start:w_start+side_len]
            new_patch = torch.clamp(current_patch + perturbation, 0.0, 255.0)
            x_try[:, :, h_start:h_start+side_len, w_start:w_start+side_len] = new_patch
            
            # Assicura il rispetto del bound globale di epsilon rispetto all'originale
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)
            
            current_loss = self._get_loss(x_try)
            
            # Se la loss diminuisce (il detector rileva meno o con minor confidenza), accettiamo la modifica
            if current_loss < best_loss:
                best_loss = current_loss
                x_best = x_try
                
                if best_loss == 0.0:
                    return x_best, True, queries

        # Controllo finale del successo dell'evasione
        final_dets = self.detector_wrapper(x_best.squeeze(0).byte().float())
        success = (final_dets is None or len(final_dets) == 0 or len(final_dets[0]) == 0)

        return x_best, success, queries
