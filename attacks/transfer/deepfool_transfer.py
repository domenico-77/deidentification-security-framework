import torch
import torch.nn as nn

class TransferDeepFoolAttack:
    """
    Implementazione di DeepFool per Transfer / Gray-Box Attacks,
    sfruttando un modello surrogato (YOLO o RetinaFace) per il calcolo iterativo
    della perturbazione minima e verificando l'evasione sul target (DSFD).
    """
    def __init__(
        self, 
        detector=None, 
        detector_wrapper=None, 
        dsfd_net=None, 
        mean_tensor=None, 
        surrogate_net=None, 
        device: torch.device = None,
        max_iter=20,
        overshoot=0.02
    ):
        self.detector = detector if detector is not None else detector_wrapper
        self.dsfd_net = dsfd_net.to(device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(device) if mean_tensor is not None else None
        self.surrogate_net = surrogate_net.to(device) if surrogate_net is not None else None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iter = max_iter
        self.overshoot = overshoot

    def perturb(self, img_orig_tensor, epsilon=8.0):
        """
        Esegue l'attacco DeepFool iterativo basato sul surrogato.
        Nota: 'epsilon' viene mantenuto per compatibilità di firma con la factory,
        ma DeepFool calcola dinamicamente la perturbazione minima.
        """
        if img_orig_tensor.ndim == 3:
            img_orig_tensor = img_orig_tensor.unsqueeze(0)

        img_adv = img_orig_tensor.clone().detach().to(self.device).float()
        img_orig_tensor = img_orig_tensor.to(self.device).float()
        
        success = False
        success_iteration = None

        if self.surrogate_net is None:
            raise ValueError("DeepFool Transfer richiede un modello surrogato valido.")

        for i in range(self.max_iter):
            img_adv.requires_grad_(True)
            
            # Surrogato richiede 4D [B, C, H, W] in [0, 1]
            input_surrogate = img_adv / 255.0 if img_adv.max() > 1.0 else img_adv
            if input_surrogate.ndim == 3:
                input_surrogate = input_surrogate.unsqueeze(0)

            # Calcolo della loss di evasione sul surrogato
            if hasattr(self.surrogate_net, "compute_adversarial_loss"):
                loss = self.surrogate_net.compute_adversarial_loss(input_surrogate)
            else:
                surrogate_out = self.surrogate_net(input_surrogate)
                loss = torch.relu(surrogate_out[1]).sum() if isinstance(surrogate_out, (list, tuple)) else torch.tensor(0.0, device=self.device, requires_grad=True)

            self.surrogate_net.zero_grad()
            if img_adv.grad is not None:
                img_adv.grad.zero_()
                
            loss.backward()

            if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
                break

            grad = img_adv.grad.clone()
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1)
            grad_norm = torch.clamp(grad_norm, min=1e-8)

            # Calcolo del passo DeepFool linearizzato
            # r_i = (|loss| / ||grad||^2) * grad
            value = torch.abs(loss)
            r_i = (value / (torch.sum(grad ** 2) + 1e-8)) * grad
            
            with torch.no_grad():
                img_adv = img_adv + (1.0 + self.overshoot) * r_i
                # Clipping nel range valido [0, 255] e opzionalmente nel budget epsilon se specificato
                eta = img_adv - img_orig_tensor
                eps_val = epsilon / 255.0 if epsilon > 1.0 else epsilon
                # Se si vuole rispettare strict L_inf budget:
                # eta = torch.clamp(eta, min=-eps_val * 255.0, max=eps_val * 255.0)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

            # Verifica dell'evasione sul detector target reale (DSFD) -> richiede 3D [C, H, W]
            detector_input = img_adv.detach().byte().float()
            if detector_input.ndim == 4:
                detector_input = detector_input[0]
                
            with torch.no_grad():
                detections = self.detector(detector_input)
            
            chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
            if chk_faces == 0:
                success = True
                success_iteration = i + 1
                break

        if not success:
            success_iteration = self.max_iter

        return img_adv[0] if img_adv.ndim == 4 else img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
