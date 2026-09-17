import torch
from attacks.base_attack import BaseAttack

class PGDAttack(BaseAttack):
    def __init__(self, detector, dsfd_net, mean_tensor, device: torch.device = None):
        super().__init__(detector=detector, device=device)
        self.dsfd_net = dsfd_net.to(self.device)
        self.mean_tensor = mean_tensor.to(self.device)

    def perturb(self, img_orig_tensor, epsilon, alpha=2.0, iterations=150):
        """
        Esegue l'attacco iterativo PGD mirato a eludere il rilevatore DSFD.
        Restituisce: (img_adv, success, success_iteration)
        """
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        success = False
        success_iteration = None

        for i in range(iterations):
            img_adv.requires_grad_(True)
            
            # 1. Input normalizzato per DSFD
            input_net = img_adv.unsqueeze(0) - self.mean_tensor
            
            # 2. Forward pass
            net_out = self.dsfd_net(input_net, 0.0, 0.0)
            
            # 3. Calcolo Loss di evasione
            loss = 0.0
            if isinstance(net_out, (list, tuple)):
                for t in net_out:
                    if isinstance(t, torch.Tensor) and t.requires_grad:
                        if t.ndim >= 2 and t.shape[-1] == 2:
                            face_logits = t[..., 1]
                            bg_logits = t[..., 0]
                            loss = loss + torch.relu(face_logits - bg_logits).sum()
                        else:
                            loss = loss + torch.relu(t).sum()
            elif isinstance(net_out, torch.Tensor) and net_out.requires_grad:
                loss = torch.relu(net_out).sum()

            self.dsfd_net.zero_grad()
            loss.backward()

            if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
                break

            grad_sign = img_adv.grad.sign()

            # 4. Aggiornamento PGD e Proiezione L-inf su scala [0, 255]
            with torch.no_grad():
                img_adv = img_adv - alpha * grad_sign
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

            # 5. Check Evasione (ogni 5 iterazioni)
            if i % 5 == 0 or i == iterations - 1:
                detector_input = img_adv.detach().byte().float()
                with torch.no_grad():
                    detections = self.detector(detector_input)
                
                chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
                
                if chk_faces == 0:
                    success = True
                    success_iteration = i
                    break

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        """
        Implementazione del metodo astratto richiesto da BaseAttack.
        Restituisce direttamente il tensore adversarial.
        """
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
