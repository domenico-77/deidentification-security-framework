import torch
from attacks.fgsm import FGSMAttack

class TransferFGSMAttack(FGSMAttack):
    """
    Implementazione dell'attacco FGSM in modalità transfer,
    che eredita da FGSMAttack, calcola i gradienti sul modello surrogato (es. YOLO)
    e verifica il successo dell'evasione sul detector target reale.
    """
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, surrogate_net=None, device: torch.device = None, **kwargs):
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def perturb(self, img_orig_tensor, epsilon):
        img_adv = img_orig_tensor.clone().detach().to(self.device).float()
        img_orig_tensor = img_orig_tensor.to(self.device).float()
        
        # Normalizzazione in scala [0, 1] se necessario per il surrogato
        if img_adv.max() > 1.0:
            img_adv = img_adv / 255.0
            img_orig_tensor = img_orig_tensor / 255.0

        img_adv.requires_grad_(True)
        
        # Azzeramento gradienti del modello surrogato
        if hasattr(self.surrogate_net, 'zero_grad'):
            self.surrogate_net.zero_grad()

        # Calcolo della loss avversaria tramite il modello surrogato (es. feature divergence)
        loss = self.surrogate_net.compute_adversarial_loss(img_adv.unsqueeze(0) if img_adv.dim() == 3 else img_adv)

        # Backward pass per calcolare il gradiente rispetto all'input
        if img_adv.grad is not None:
            img_adv.grad.zero_()
            
        loss.backward()

        success = False
        success_iteration = None
        epsilon_norm = epsilon / 255.0 if epsilon > 1.0 else epsilon

        # Applicazione del passo FGSM sul surrogato
        if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
            grad_sign = img_adv.grad.sign()
            if grad_sign.dim() == 5 and img_adv.dim() == 4:
                grad_sign = grad_sign.mean(dim=1)
                
            with torch.no_grad():
                img_adv = img_adv - epsilon_norm * grad_sign
                
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon_norm, max=epsilon_norm)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=1.0)
                
            # Ri-portiamo l'immagine in scala [0, 255] per il controllo sul detector target
            img_adv_eval = img_adv * 255.0

            # Verifica dell'evasione sul DETECTOR TARGET REALE (DSFD)
            with torch.no_grad():
                detector_input = img_adv_eval.detach().byte().float()
                detections = self.detector(detector_input)
                chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
                if chk_faces == 0:
                    success = True
                    success_iteration = 1

        # Riconversione finale in scala [0, 255] coerente con il framework
        if img_adv.max() <= 1.0:
            img_adv = img_adv * 255.0

        return img_adv.detach(), success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
