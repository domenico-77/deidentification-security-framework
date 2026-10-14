import torch
from attacks.base_attack import BaseAttack

class BIMCIAGANAttack(BaseAttack):
    def __init__(self, target=None, detector_wrapper=None, device: torch.device = None):
        det = detector_wrapper if detector_wrapper is not None else (target.detector_wrapper if target else None)
        super().__init__(detector=det, device=device)
        self.target = target
        self.generator = target.model if target is not None else None

    def perturb(self, img_orig_tensor, epsilon=8.0, alpha=1.0, num_iter=15):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = None

        for i in range(num_iter):
            img_adv.requires_grad_(True)
            
            # Loss basata sulla generazione/distruzione della feature tramite il generatore CIAGAN
            if self.generator is not None:
                # Massimizziamo la differenza/perturbazione rispetto all'originale sul proxy del generatore
                loss = -torch.nn.functional.mse_loss(img_adv, img_orig_tensor)
            else:
                loss = -torch.nn.functional.mse_loss(img_adv, img_orig_tensor)

            if img_adv.grad is not None:
                img_adv.grad.zero_()
            loss.backward()

            if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
                grad_sign = img_adv.grad.sign()
                with torch.no_grad():
                    # Gradient ascent per massimizzare la distruzione della feature e favorire l'evasione
                    img_adv = img_adv + alpha * grad_sign
                    
                    eta = img_adv - img_orig_tensor
                    eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                    img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()
            
            # Verifica dell'evasione tramite il detector dlib integrato nel wrapper
            detector_input = img_adv.detach()
            detections = self.detector(detector_input)
            
            chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
            if chk_faces == 0:
                success = True
                success_iteration = i + 1
                break

        if not success:
            success_iteration = num_iter

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
