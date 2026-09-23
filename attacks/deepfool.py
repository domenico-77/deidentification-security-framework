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
        x = img_orig_tensor.clone().detach().to(self.device).float()
        x.requires_grad = True
        
        original_image = x.clone()
        x_orig_np = img_orig_tensor.detach().cpu().numpy()

        success = False
        succ_iter = 0

        for i in range(max_iter):
            x_norm = x - self.mean_tensor
            
            # CORREZIONE: Passaggio dei parametri obbligatori a DSFD (confidence_threshold e nms_threshold)
            # Di solito i valori di default sono 0.5 per la confidenza e 0.45 o 0.3 per la NMS
            try:
                outputs = self.dsfd_net(x_norm, confidence_threshold=0.5, nms_threshold=0.4)
            except TypeError:
                # Fallback nel caso la firma accetti argomenti diversi o sia il wrapper
                outputs = self.dsfd_net(x_norm)

            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                # Somma dei tensori di output per stimare il livello di attivazione/confidenza
                score = sum([o.sum() for o in outputs if isinstance(o, torch.Tensor)])
            elif isinstance(outputs, torch.Tensor):
                score = outputs.sum()
            else:
                score = x_norm.sum() # Fallback di sicurezza

            # Condizione di evasione (se il punteggio crolla o non rileva più nulla)
            if score.item() < 0.0:
                success = True
                succ_iter = i + 1
                break

            self.dsfd_net.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
                
            score.backward()
            grad = x.grad.data.clone()

            w = grad
            f_x = score

            if torch.norm(w) == 0:
                break

            pert = (torch.abs(f_x) / (torch.norm(w) ** 2 + 1e-8)) * w * (1 + overshoot)
            
            with torch.no_grad():
                x += pert
                x = torch.clamp(x, 0, 255)
                x.requires_grad = True

            succ_iter = i + 1

        img_adv = x.detach()
        
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
