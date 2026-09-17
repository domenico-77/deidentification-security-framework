import torch
from attacks.base_attack import BaseAttack

class FGSMAttack(BaseAttack):
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, device: torch.device = None):
        det = detector if detector is not None else detector_wrapper
        super().__init__(detector=det, device=device)
        
        self.detector_wrapper = self.detector
        self.dsfd_net = dsfd_net.to(self.device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(self.device) if mean_tensor is not None else None

    def perturb(self, img_orig_tensor, epsilon):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        # Abilitiamo i gradienti per il singolo passo
        img_adv.requires_grad_(True)
        
        # 1. Normalizzazione dell'input per il DSFD
        input_net = img_adv.unsqueeze(0) - self.mean_tensor
        
        # 2. Forward pass per estrarre i logit grezzi della rete di detection
        net_out = self.dsfd_net(input_net, 0.0, 0.0)
        
        # 3. Loss di massimizzazione mirata (Targeted Logit Inversion)
        # Vogliamo massimizzare il logit dello sfondo (t[..., 0]) e minimizzare quello della faccia (t[..., 1])
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if isinstance(net_out, (list, tuple)):
            for t in net_out:
                if isinstance(t, torch.Tensor):
                    if t.ndim >= 2 and t.shape[-1] == 2:
                        # Differenza direzionale netta orientata all'inganno del classificatore
                        loss = loss + (t[..., 1] - t[..., 0]).sum()
                    else:
                        loss = loss + t.sum()
        elif isinstance(net_out, torch.Tensor):
            loss = loss + net_out.sum()

        # 4. Backward pass per calcolare il gradiente in un unico colpo
        self.dsfd_net.zero_grad()
        loss.backward()

        success = False
        success_iteration = 1

        # 5. Applicazione del passo singolo massimizzato (FGSM Puro)
        if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
            grad_sign = img_adv.grad.sign()
            with torch.no_grad():
                # Balzo secco di ampiezza epsilon nella direzione del gradiente
                img_adv = img_adv + epsilon * grad_sign
                
                # Vincolo di clipping L-infinito per rispettare il budget di epsilon
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()
                
        # 6. Verifica dell'effettiva evasione del detector
        detector_input = img_adv.detach().byte().float()
        with torch.no_grad():
            detections = self.detector(detector_input)
        
        chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
        if chk_faces == 0:
            success = True

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
