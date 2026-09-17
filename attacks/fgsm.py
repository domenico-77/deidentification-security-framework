import torch
from attacks.base_attack import BaseAttack

class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method (FGSM) Single-step attack."""

    def __init__(self, detector_wrapper, dsfd_net, mean_tensor, device="cuda"):
        super().__init__()
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net
        self.mean_tensor = mean_tensor
        self.device = device

    def compute_adversarial_loss(self, img_tensor):
        """Calcola la loss basata sulla risposta del detector DSFD."""
        # Normalizzazione attesa dal detector DSFD di DeepPrivacy2
        normalized_img = img_tensor - self.mean_tensor
        
        # Esecuzione del forward pass sul detector in modalità training per preservare il grafo dei gradienti
        # Nota: a seconda di come è esposta la rete DSFD nel tuo wrapper, potresti chiamare direttamente self.dsfd_net
        detections = self.dsfd_net(normalized_img)
        
        # La loss punta a minimizzare la confidenza o il numero di oggetti rilevati
        # Una loss comune per l'evasione del detector è la somma dei punteggi di confidenza rilevati
        loss = 0.0
        if isinstance(detections, (list, tuple)):
            for det in detections:
                if det is not None and len(det) > 0:
                    # Somma delle confidenze delle bbox rilevate
                    loss = loss + det[:, 4].sum()
        elif torch.is_tensor(detections) and detections.numel() > 0:
            loss = detections[:, 4].sum()
        else:
            # Fallimento fittizio se non ci sono tensor lossabili, usa una norma di appoggio o dummy loss
            loss = torch.sum(img_tensor * 0.0)
            
        return loss

    def perturb(self, img_orig_tensor: torch.Tensor, epsilon: float = 32.0):
        """
        Esegue l'attacco FGSM in un unico step.
        Restituisce: (img_adv, success_bool, iterations_count)
        """
        if img_orig_tensor.dim() == 3:
            orig_tensor = img_orig_tensor.unsqueeze(0).to(self.device).float().clone()
        else:
            orig_tensor = img_orig_tensor.to(self.device).float().clone()

        orig_tensor.requires_grad = True
        
        # Calcolo della loss
        loss = self.compute_adversarial_loss(orig_tensor)

        if loss is not None and torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            if orig_tensor.grad is not None:
                grad_sign = orig_tensor.grad.sign()
                # Sottrazione per minimizzare la confidenza del detector (attacco d'evasione)
                adv_tensor = orig_tensor - epsilon * grad_sign
                # Clamping nei limiti validi dei pixel [0, 255]
                adv_tensor = torch.clamp(adv_tensor, 0.0, 255.0)
                
                # Verifica rapida se ha eluso il detector
                with torch.no_grad():
                    eval_input = adv_tensor.byte().float()
                    final_dets = self.detector_wrapper(eval_input)
                    success = (len(final_dets) == 0 or final_dets[0] is None or len(final_dets[0]) == 0)
                
                return adv_tensor.squeeze(0), success, 1

        # Fallback se il gradiente non è disponibile
        return orig_tensor.squeeze(0), False, 1
