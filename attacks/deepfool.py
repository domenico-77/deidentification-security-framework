import torch
import numpy as np
from attacks.base_attack import BaseAttack

class DeepFoolAttack(BaseAttack):
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, device: torch.device = None):
        det = detector if detector is not None else detector_wrapper
        super().__init__(detector=det, device=device)
        
        self.detector_wrapper = self.detector
        self.dsfd_net = dsfd_net.to(self.device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(self.device) if mean_tensor is not None else None

    def perturb(self, img_orig_tensor, max_iter=50, overshoot=0.02):
        """
        Esegue l'attacco DeepFool per trovare la perturbazione minima necessaria
        a evadere il rilevatore DSFD.
        """
        img_adv = img_orig_tensor.clone().detach().to(self.device).float()
        img_orig = img_orig_tensor.clone().detach().to(self.device).float()
        
        success = False
        success_iteration = None
        
        # Verifica preliminare: se già non rileva nulla, l'attacco è trivialmente riuscito
        with torch.no_grad():
            initial_det = self.detector((img_adv.unsqueeze(0)).byte().float())
            if not initial_det or initial_det[0] is None or len(initial_det[0]) == 0:
                return img_adv, True, 0

        for i in range(max_iter):
            img_adv.requires_grad_(True)
            
            # 1. Normalizzazione per DSFD
            input_net = img_adv.unsqueeze(0) - self.mean_tensor
            
            # 2. Forward pass
            net_out = self.dsfd_net(input_net, 0.0, 0.0)
            
            # Costruzione di una loss scalare basata sui logit dei volti rilevati
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if isinstance(net_out, (list, tuple)):
                for t in net_out:
                    if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[-1] == 2:
                        face_logits = t[..., 1]
                        bg_logits = t[..., 0]
                        loss = loss + torch.relu(face_logits - bg_logits).sum()
                    elif isinstance(t, torch.Tensor):
                        loss = loss + torch.relu(t).sum()
            elif isinstance(net_out, torch.Tensor):
                loss = loss + torch.relu(net_out).sum()

            if loss.item() == 0.0:
                success = True
                success_iteration = i
                break

            self.dsfd_net.zero_grad()
            if img_adv.grad is not None:
                img_adv.grad.zero_()
                
            loss.backward()
            
            if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
                break

            grad = img_adv.grad.detach()
            
            # Calcolo del passo DeepFool basato sulla linearizzazione
            w = grad
            w_norm = torch.norm(w.flatten())
            if w_norm == 0:
                break
                
            r = (abs(loss.item()) / (w_norm ** 2)) * w
            
            with torch.no_grad():
                # Applicazione del passo con overshoot per superare il confine
                img_adv = img_adv + (1 + overshoot) * r
                # Clamp sui valori validi dell'immagine [0, 255]
                img_adv = torch.clamp(img_adv, min=0.0, max=255.0).detach()

            # 3. Controllo effettivo di evasione tramite il detector wrapper
            with torch.no_grad():
                detections = self.detector(img_adv.unsqueeze(0).byte().float())
                chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
                
                if chk_faces == 0:
                    success = True
                    success_iteration = i
                    break

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, **kwargs)
        return img_adv
