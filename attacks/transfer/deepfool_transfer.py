import torch
from attacks.deepfool import DeepFoolAttack

class TransferDeepFoolAttack(DeepFoolAttack):
    """
    Versione estesa di DeepFool per attacchi di Transfer / Gray-Box,
    sfruttando un modello surrogato (YOLO o RetinaFace) per il calcolo iterativo
    della perturbazione e verificando l'evasione sul detector target (DSFD).
    """
    def __init__(
        self, 
        detector=None, 
        detector_wrapper=None, 
        dsfd_net=None, 
        mean_tensor=None, 
        surrogate_net=None, 
        device: torch.device = None
    ):
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def perturb(self, img_orig_tensor, max_iter=50, overshoot=0.02, epsilon=8.0, **kwargs):
        """
        Esegue l'attacco DeepFool Transfer basato sul surrogato,
        verificando l'evasione sul detector target (DSFD).
        """
        if img_orig_tensor.ndim == 3:
            img_orig_tensor = img_orig_tensor.unsqueeze(0)

        x = img_orig_tensor.clone().detach().to(self.device).float()
        original_image = x.clone()

        success = False
        succ_iter = 0

        # Se non è presente un modello surrogato, esegue il DeepFool standard sul target
        if self.surrogate_net is None:
            return super().perturb(img_orig_tensor.squeeze(0), max_iter=max_iter, overshoot=overshoot, **kwargs)

        for i in range(max_iter):
            x.requires_grad = True

            # 1. Surrogato richiede 4D [B, C, H, W] in [0, 1]
            input_surrogate = x / 255.0 if x.max() > 1.0 else x
            if input_surrogate.ndim == 3:
                input_surrogate = input_surrogate.unsqueeze(0)

            # 2. Calcolo Loss di evasione tramite il metodo standardizzato del surrogato
            if hasattr(self.surrogate_net, "compute_adversarial_loss"):
                loss = self.surrogate_net.compute_adversarial_loss(input_surrogate)
            else:
                surrogate_out = self.surrogate_net(input_surrogate)
                loss = torch.relu(surrogate_out[1]).sum() if isinstance(surrogate_out, (list, tuple)) else torch.tensor(0.0, device=self.device, requires_grad=True)

            # 3. Backward pass sul surrogato
            self.surrogate_net.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            loss.backward()

            if x.grad is None or torch.abs(x.grad).sum().item() == 0:
                break

            grad = x.grad.data.clone()
            w = grad
            f_x = loss

            if torch.norm(w) == 0:
                break

            scaling_factor = 0.05
            pert = (torch.abs(f_x) / (torch.norm(w) ** 2 + 1e-8)) * w * (1 + overshoot) * scaling_factor

            with torch.no_grad():
                x = x + pert
                # Opzionale: rispetto del budget L-inf di epsilon se desiderato
                eta = x - original_image
                eps_val = epsilon / 255.0 if epsilon > 1.0 else epsilon
                eta = torch.clamp(eta, min=-eps_val * 255.0, max=eps_val * 255.0)
                x = torch.clamp(original_image + eta, min=0.0, max=255.0).detach()

            # 4. Verifica dell'evasione sul detector target reale (DSFD) -> richiede 3D [C, H, W]
            detector_input = x.detach().byte().float()
            if detector_input.ndim == 4:
                detector_input = detector_input[0]

            with torch.no_grad():
                try:
                    dets = self.detector_wrapper(detector_input)
                    face_detected = (dets is not None and len(dets) > 0 and len(dets[0]) > 0)
                except Exception:
                    face_detected = True

            if not face_detected:
                success = True
                succ_iter = i + 1
                break

            succ_iter = i + 1

        img_adv = x.detach()

        return img_adv[0] if img_adv.ndim == 4 else img_adv, success, succ_iter

    def attack(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, **kwargs)
        return img_adv
