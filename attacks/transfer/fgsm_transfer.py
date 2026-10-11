import torch
from attacks.fgsm import FGSMAttack

class TransferFGSMAttack(FGSMAttack):
    """
    Versione estesa di FGSM per attacchi di Transfer / Gray-Box,
    sfruttando un modello surrogato (YOLO o RetinaFace) per il calcolo dei gradienti.
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

    def perturb(self, img_orig_tensor, epsilon):
        """
        Esegue l'attacco FGSM Transfer basato sul surrogato,
        verificando l'evasione sul detector target (DSFD).
        """
        if img_orig_tensor.ndim == 3:
            img_orig_tensor = img_orig_tensor.unsqueeze(0)

        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        success = False
        success_iteration = None

        # Se non è presente un modello surrogato, esegue l'FGSM standard sul target
        if self.surrogate_net is None:
            return super().perturb(img_orig_tensor, epsilon=epsilon)

        img_adv.requires_grad_(True)

        # 1. Adatta l'input in base al modello surrogato (es. normalizzazione [0, 1])
        input_surrogate = img_adv / 255.0 if img_adv.max() > 1.0 else img_adv

        # 2. Calcolo Loss di evasione tramite il metodo standardizzato del surrogato
        if hasattr(self.surrogate_net, "compute_adversarial_loss"):
            loss = self.surrogate_net.compute_adversarial_loss(input_surrogate)
        else:
            surrogate_out = self.surrogate_net(input_surrogate)
            loss = torch.relu(surrogate_out[1]).sum() if isinstance(surrogate_out, (list, tuple)) else torch.tensor(0.0, device=self.device, requires_grad=True)

        # 3. Backward pass sul surrogato
        self.surrogate_net.zero_grad()
        if img_adv.grad is not None:
            img_adv.grad.zero_()
            
        loss.backward()

        # 4. Applicazione del passo FGSM e proiezione L-inf
        if img_adv.grad is not None and torch.abs(img_adv.grad).sum().item() > 0:
            grad_sign = img_adv.grad.sign()
            with torch.no_grad():
                img_adv = img_adv - epsilon * grad_sign
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

            # 5. Verifica dell'evasione sul detector target reale (DSFD) assicurando input 4D
            detector_input = img_adv.detach().byte().float()
            if detector_input.ndim == 3:
                detector_input = detector_input.unsqueeze(0)
                
            with torch.no_grad():
                detections = self.detector(detector_input)
            
            chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
            if chk_faces == 0:
                success = True
                success_iteration = 1

        return img_adv, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=8.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
