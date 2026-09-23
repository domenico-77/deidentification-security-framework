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
            
            # 1. Eseguiamo il detector reale sull'immagine corrente per vedere se il volto è ancora rilevato
            with torch.no_grad():
                eval_input = x.byte().float()
                # Usa il wrapper del detector per ottenere le box attuali
                try:
                    dets = self.detector_wrapper(eval_input)
                    face_detected = (dets is not None and len(dets) > 0 and len(dets[0]) > 0)
                except Exception:
                    face_detected = True

            # Se il detector NON rileva più il volto, l'attacco DeepFool ha SUCCESSO!
            if not face_detected:
                success = True
                succ_iter = i + 1
                break

            # 2. Calcolo dei gradienti per spingere l'ottimizzazione verso l'evasione
            outputs = self.dsfd_net(x_norm, confidence_threshold=0.5, nms_threshold=0.4)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                score = sum([o.sum() for o in outputs if isinstance(o, torch.Tensor)])
            elif isinstance(outputs, torch.Tensor):
                score = outputs.sum()
            else:
                score = x_norm.sum()

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
