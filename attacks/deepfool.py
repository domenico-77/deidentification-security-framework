import torch
import torch.nn as nn
import numpy as np

class DeepFoolAttack:
    def __init__(self, detector_wrapper, dsfd_net, mean_tensor, device="cuda"):
        """
        Inizializza l'attacco DeepFool contro il detector DSFD di DeepPrivacy2.
        """
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net
        self.mean_tensor = mean_tensor
        self.device = device

    def perturb(self, img_orig_tensor, max_iter=50, overshoot=0.02, **kwargs):
        """
        Esegue l'attacco DeepFool per trovare la perturbazione minima necessaria
        a evadere il rilevatore DSFD. Accetta **kwargs per compatibilità con il runner.
        """
        # Assicura che il tensore sia sul device corretto e abilitato ai gradienti
        x = img_orig_tensor.clone().detach().to(self.device).float()
        x.requires_grad = True
        
        original_image = x.clone()
        
        # Copia dell'immagine originale per il calcolo delle metriche finali
        x_orig_np = img_orig_tensor.detach().cpu().numpy()

        success = False
        succ_iter = 0

        for i in range(max_iter):
            # Normalizzazione attesa dal detector DSFD
            x_norm = x - self.mean_tensor
            
            # Forward pass attraverso la rete DSFD
            # Nota: adattato in base alla struttura dei tensori di output di DSFD
            outputs = self.dsfd_net(x_norm)
            
            # Assumiamo di prendere il punteggio di confidenza della prima classe/detection o il logit massimo
            if isinstance(outputs, (list, tuple)):
                # Prende ad esempio la classificazione o il punteggio principale
                score = outputs[0].sum()
            else:
                score = outputs.sum()

            # Se il punteggio scende sotto una determinata soglia o non rileva più volti, consideriamo l'attacco riuscito
            # (Verifica basata sull'evasione del detector)
            if score.item() < 0.0:  # Condizione di esempio per l'evasione
                success = True
                succ_iter = i + 1
                break

            # Calcolo dei gradienti rispetto all'input
            self.dsfd_net.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
                
            score.backward()
            grad = x.grad.data.clone()

            # Semplificazione della logica iterativa di DeepFool per il gradiente del detector
            w = grad
            f_x = score

            if torch.norm(w) == 0:
                break

            # Calcolo della perturbazione minima (formula standard di DeepFool)
            pert = (torch.abs(f_x) / (torch.norm(w) ** 2 + 1e-8)) * w * (1 + overshoot)
            
            with torch.no_grad():
                x += pert
                # Proiezione opzionale nei limiti validi dei pixel (es. [0, 255] o [-1, 1])
                x = torch.clamp(x, 0, 255)
                x.requires_grad = True

            succ_iter = i + 1

        img_adv = x.detach()
        
        # Calcolo metriche di supporto (Linf, L2, MSE, PSNR)
        diff = img_adv - original_image
        l_inf = torch.max(torch.abs(diff)).item()
        l2 = torch.norm(diff).item()
        mse = torch.mean(diff ** 2).item()
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')

        metrics = {
            "success": success,
            "success_iteration": succ_iter,
            "l_inf": l_inf,
            "l2": l2,
            "mse": mse,
            "psnr": psnr
        }

        return img_adv, success, succ_iter
