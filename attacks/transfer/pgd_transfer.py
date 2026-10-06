import torch
import torch.nn.functional as F
from attacks.pgd import PGDAttack

class TransferPGDAttack(PGDAttack):
    """
    Versione estesa di PGD per attacchi di Transfer / Gray-Box (es. utilizzando un modello surrogato come YOLO o RetinaFace).
    Eredita da PGDAttack mantenendo intatta la classe originale.
    """
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, surrogate_net=None, device: torch.device = None):
        # Inizializziamo la classe base
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        
        # Modello surrogato alternativo per il trasferimento dei gradienti
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def perturb(self, img_orig_tensor, epsilon, alpha=2.0, iterations=150):
        """
        Esegue il PGD basato sul modello surrogato (Transfer Attack), proiettando 
        la perturbazione nel budget epsilon e verificando l'evasione sul DSFD originale.
        """
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        success = False
        success_iteration = None

        # Se non è stato fornito un modello surrogato, esegue il PGD standard della classe padre
        if self.surrogate_net is None:
            return super().perturb(img_orig_tensor, epsilon=epsilon, alpha=alpha, iterations=iterations)

        for i in range(iterations):
            img_adv.requires_grad_(True)
            
            # 1. Forward pass sul modello surrogato (es. YOLO o altro detector)
            # Adatta l'input in base al modello surrogato (es. normalizzazione [0, 1] per YOLO)
            input_surrogate = img_adv.unsqueeze(0) / 255.0
            
            surrogate_out = self.surrogate_net(input_surrogate)
            
            # 2. Calcolo Loss di evasione basata sull'output del surrogato
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Gestione flessibile in base al tipo di output del surrogato (es. YOLO boxes / logit generici)
            if isinstance(surrogate_out, (list, tuple)):
                for t in surrogate_out:
                    if isinstance(t, torch.Tensor):
                        loss = loss + torch.relu(t).sum()
            elif hasattr(surrogate_out, "boxes") and surrogate_out.boxes is not None:
                # Gestione specifica per YOLOv8 / Ultralytics se restituito come dizionario/risultato
                boxes = surrogate_out.boxes
                if boxes.conf is not None and len(boxes.conf) > 0:
                    loss = boxes.conf.sum()
                else:
                    loss = img_adv.sum() * 0.0
            elif isinstance(surrogate_out, torch.Tensor):
                loss = torch.relu(surrogate_out).sum()
            else:
                loss = img_adv.sum() * 0.0

            # 3. Backward pass sul surrogato
            self.surrogate_net.zero_grad()
            if img_adv.grad is not None:
                img_adv.grad.zero_()
                
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

            # 5. Check Evasione sul detector target reale (DSFD / DeepPrivacy2) ogni 5 iterazioni
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
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
