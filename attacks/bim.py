import torch
from attacks.base_attack import BaseAttack

class BIMAttack(BaseAttack):
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, device: torch.device = None):
        det = detector if detector is not None else detector_wrapper
        super().__init__(detector=det, device=device)
        
        self.detector_wrapper = self.detector
        self.dsfd_net = dsfd_net.to(self.device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(self.device) if mean_tensor is not None else None

    def perturb(self, img_orig_tensor, epsilon=8.0, alpha=1.0, num_iter=15):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = None

        # Ciclo iterativo tipico del BIM (Basic Iterative Method)
        for i in range(num_iter):
            img_adv.requires_grad_(True)
            
            # 1. Normalizzazione dell'input per il DSFD
            input_net = img_adv.unsqueeze(0) - self.mean_tensor
            
            # 2. Forward pass per estrarre i logit grezzi
            net_out = self.dsfd_net(input_net, 0.0, 0.0)
            
            # 3. Loss di evasione (penalizza la confidenza del volto)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if isinstance(net_out, (list, tuple)):
                for t in net_out:
                    if isinstance(t, torch.Tensor):
                        if t.ndim >= 2 and t.shape[-1] == 2:
                            face_logits = t[..., 1]
                            bg_logits = t[..., 0]
                            loss = loss + torch.relu(face_logits - bg_logits).sum()
                        else:
                            loss = loss + torch.relu(t).sum()
            elif isinstance(net_out, torch.Tensor):
                loss = loss + torch.relu(net_out).sum()

            # 4. Backward pass per calcolare il gradiente
            self.dsfd_net.zero_grad()
            if img_adv.grad is not None:
                img_adv.grad.zero_()
            loss.backward()

            # 5. Aggiornamento a piccoli passi con proiezione stretta nella epsilon-ball
            if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
                grad_sign = img_adv.grad.sign()
                with torch.no_grad():
                    # Passo incrementale nella direzione di minimizzazione della confidenza
                    img_adv = img_adv - alpha * grad_sign
                    
                    # Proiezione e clipping rigoroso nel budget epsilon e nei limiti dei pixel [0, 255]
                    eta = img_adv - img_orig_tensor
                    eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                    img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()
            
            # 6. Verifica dell'effettiva evasione del detector a ogni iterazione
            detector_input = img_adv.detach().byte().float()
            with torch.no_grad():
                detections = self.detector(detector_input)
            
            chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
            if chk_faces == 0:
                success = True
                success_iteration = i + 1
                break

        # Se fallisce, assegniamo il numero massimo di iterazioni (coerente con il benchmark runner)
        if not success:
            success_iteration = num_iter

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
