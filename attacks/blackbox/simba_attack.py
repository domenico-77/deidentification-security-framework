import torch


class SimBA:
    """
    SimBA (Simple Black-Box Attack).
    Attacco black-box basato su query che campiona direzioni casuali (es. pixel o blocchi)
    e testa se aggiungere o sottrarre epsilon migliora l'efficacia dell'attacco (evasione del detector).
    """
    def __init__(self, detector_wrapper, device="cuda", n_queries=1000, step_size=2.0):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.n_queries = n_queries
        self.step_size = step_size

    def _get_loss(self, img_tensor):
        """
        Calcola una loss proxy basata sulla somma delle confidenze dei box rilevati.
        L'obiettivo è minimizzare questa loss per ingannare il detector.
        """
        with torch.no_grad():
            eval_input = img_tensor.squeeze(0).byte().float() if img_tensor.ndim == 4 else img_tensor.byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input[0]
            
            dets = self.detector_wrapper(eval_input)
            if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                return 0.0  # Faccia evitata con successo
            
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
        Esegue SimBA su una singola immagine.
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

        # Genera una lista di tutte le possibili direzioni ortogonali (o coordinate dei pixel)
        # Per efficienza su immagini ad alta risoluzione, possiamo campionare coordinate casuali
        for query_idx in range(1, self.n_queries + 1):
            queries = query_idx

            # Seleziona una direzione casuale (un canale e un punto/blocco (h, w))
            c_idx = torch.randint(0, c, (1,)).item()
            h_idx = torch.randint(0, h, (1,)).item()
            w_idx = torch.randint(0, w, (1,)).item()

            # Crea il vettore di direzione ortogonale e_i
            direction = torch.zeros((1, c, h, w), device=self.device)
            direction[0, c_idx, h_idx, w_idx] = 1.0

            # Prova a sottrarre il passo
            x_try = x_best - self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_sub = self._get_loss(x_try)

            if loss_sub < best_loss:
                best_loss = loss_sub
                x_best = x_try
                if best_loss == 0.0:
                    return x_best, True, queries
                continue

            # Se sottrarre non migliora, prova a sommare il passo
            x_try = x_best + self.step_size * direction
            x_try = torch.clamp(x_try, x_orig - epsilon, x_orig + epsilon)
            x_try = torch.clamp(x_try, 0.0, 255.0)

            loss_add = self._get_loss(x_try)

            if loss_add < best_loss:
                best_loss = loss_add
                x_best = x_try
                if best_loss == 0.0:
                    return x_best, True, queries

        # Controllo finale del successo dell'evasione
        final_dets = self.detector_wrapper(x_best.squeeze(0).byte().float())
        success = (final_dets is None or len(final_dets) == 0 or len(final_dets[0]) == 0)

        return x_best, success, queries
