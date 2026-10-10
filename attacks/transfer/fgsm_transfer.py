import torch
from attacks.fgsm import FGSMAttack

class TransferFGSMAttack(FGSMAttack):
    """
    Implementazione dell'attacco FGSM in modalità transfer,
    con gestione unificata dell'output del surrogato (YOLO / RetinaFace).
    """
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, surrogate_net=None, device: torch.device = None, **kwargs):
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def perturb(self, img_orig_tensor, epsilon):
        img_adv = img_orig_tensor.clone().detach().to(self.device).float()
        img_orig_tensor = img_orig_tensor.to(self.device).float()
        
        if img_adv.max() > 1.0:
            img_adv = img_adv / 255.0
            img_orig_tensor = img_orig_tensor / 255.0

        img_adv.requires_grad_(True)
        
        if hasattr(self.surrogate_net, 'zero_grad'):
            self.surrogate_net.zero_grad()

        input_surrogate = img_adv.unsqueeze(0) if img_adv.dim() == 3 else img_adv
        surrogate_out = self.surrogate_net(input_surrogate)

        # Calcolo della loss unificato (uguale a PGD)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if isinstance(surrogate_out, (list, tuple)):
            for t in surrogate_out:
                if isinstance(t, torch.Tensor):
                    loss = loss + torch.relu(t).sum()
        elif hasattr(surrogate_out, "boxes") and surrogate_out.boxes is not None:
            boxes = surrogate_out.boxes
            if boxes.conf is not None and len(boxes.conf) > 0:
                loss = boxes.conf.sum()
            else:
                loss = img_adv.sum() * 0.0
        elif isinstance(surrogate_out, torch.Tensor):
            loss = torch.relu(surrogate_out).sum()
        else:
            loss = img_adv.sum() * 0.0

        if img_adv.grad is not None:
            img_adv.grad.zero_()
            
        loss.backward()

        success = False
        success_iteration = None
        epsilon_norm = epsilon / 255.0 if epsilon > 1.0 else epsilon

        if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
            grad_sign = img_adv.grad.sign()
            if grad_sign.dim() == 5 and img_adv.dim() == 4:
                grad_sign = grad_sign.mean(dim=1)
                
            with torch.no_grad():
                img_adv = img_adv - epsilon_norm * grad_sign
                
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon_norm, max=epsilon_norm)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=1.0)
                
            img_adv_eval = img_adv * 255.0

            with torch.no_grad():
                detector_input = img_adv_eval.detach().byte().float()
                detections = self.detector(detector_input)
                chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
                if chk_faces == 0:
                    success = True
                    success_iteration = 1

        if img_adv.max() <= 1.0:
            img_adv = img_adv * 255.0

        return img_adv.detach(), success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
