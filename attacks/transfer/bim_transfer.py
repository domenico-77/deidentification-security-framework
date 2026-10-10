import torch
from attacks.bim import BIMAttack

class TransferBIMAttack(BIMAttack):
    """
    Implementazione dell'attacco BIM (Basic Iterative Method) in modalità transfer,
    che eredita da BIMAttack ma sfrutta il modello surrogato (es. YOLO) 
    per il calcolo iterativo dei gradienti e delle feature divergence.
    """
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, surrogate_net=None, device: torch.device = None, **kwargs):
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def perturb(self, img_orig_tensor, epsilon=16.0, alpha=1.0, num_iter=10):
        img_adv = img_orig_tensor.clone().detach().to(self.device).float()
        img_orig_tensor = img_orig_tensor.to(self.device).float()
        
        # Normalizzazione in scala [0, 1] se necessario per il surrogato
        if img_adv.max() > 1.0:
            img_adv = img_adv / 255.0
            img_orig_tensor = img_orig_tensor / 255.0

        success = False
        success_iteration = None
        
        epsilon_norm = epsilon / 255.0 if epsilon > 1.0 else epsilon
        alpha_norm = alpha / 255.0 if alpha > 1.0 else alpha

        # Ciclo iterativo del Basic Iterative Method (BIM)
        for i in range(num_iter):
            img_adv.requires_grad_(True)
            
            # Azzeramento gradienti del modello surrogato
            if hasattr(self.surrogate_net, 'zero_grad'):
                self.surrogate_net.zero_grad()

            # Calcolo della loss avversaria tramite il modello surrogato (feature divergence)
            loss = self.surrogate_net.compute_adversarial_loss(img_adv.unsqueeze(0) if img_adv.dim() == 3 else img_adv)

            # Backward pass per calcolare il gradiente rispetto all'input
            if img_adv.grad is not None:
                img_adv.grad.zero_()
            loss.backward()

            # Aggiornamento iterativo con il segno del gradiente (BIM / Projected Gradient Descent iterativo)
            if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
                grad_sign = img_adv.grad.sign()
                if grad_sign.dim() == 5 and img_adv.dim() == 4:
                    grad_sign = grad_sign.mean(dim=1)
                    
                with torch.no_grad():
                    # Passo incrementale
                    img_adv = img_adv - alpha_norm * grad_sign
                    
                    # Proiezione rigorosa all'interno della epsilon-ball attorno all'originale
                    eta = img_adv - img_orig_tensor
                    eta = torch.clamp(eta, min=-epsilon_norm, max=epsilon_norm)
                    img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=1.0)

            # Ri-portiamo l'immagine in scala [0, 255] per il controllo sul detector target
            img_adv_eval = img_adv * 255.0

            # Verifica dell'evasione a ogni iterazione
            with torch.no_grad():
                if hasattr(self.surrogate_net, 'count_detections'):
                    det_count = self.surrogate_net.count_detections(img_adv.unsqueeze(0) if img_adv.dim() == 3 else img_adv)
                    if det_count == 0:
                        success = True
                        success_iteration = i + 1
                        break
                else:
                    detector_input = img_adv_eval.detach().byte().float()
                    detections = self.detector(detector_input)
                    chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
                    if chk_faces == 0:
                        success = True
                        success_iteration = i + 1
                        break

        if not success:
            success_iteration = num_iter

        # Riconversione finale in scala [0, 255] coerente con il framework
        if img_adv.max() <= 1.0:
            img_adv = img_adv * 255.0

        return img_adv.detach(), success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=16.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
