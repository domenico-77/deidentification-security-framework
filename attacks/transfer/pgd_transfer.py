import torch
import torch.nn.functional as F
import random
from attacks.pgd import PGDAttack

class TransferPGDAttack(PGDAttack):
    """
    Versione estesa di PGD per attacchi di Transfer / Gray-Box avanzati con 
    MI-FGSM (Momentum) e DI-FGSM (Diverse Inputs) per evitare l'overfitting sul surrogato.
    """
    def __init__(
        self, 
        detector=None, 
        detector_wrapper=None, 
        dsfd_net=None, 
        mean_tensor=None, 
        surrogate_net=None, 
        device: torch.device = None,
        use_momentum: bool = True,
        mu: float = 1.0,
        use_di: bool = True,
        di_prob: float = 0.5
    ):
        super().__init__(detector=detector, detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None
        
        # Parametri per MI-FGSM e DI-FGSM
        self.use_momentum = use_momentum
        self.mu = mu
        self.use_di = use_di
        self.di_prob = di_prob

    def diverse_inputs(self, x_tensor):
        """
        Applica trasformazioni stocastiche di input (Resize + Padding casuale) 
        per aumentare la generalizzabilità dell'attacco (DI-FGSM).
        Input atteso: batch di immagini (B, C, H, W) normalizzate o in scala [0, 255].
        """
        if not self.use_di or random.random() > self.di_prob:
            return x_tensor
        if x_tensor.dim() == 3:
            x_tensor = x_tensor.unsqueeze(0)
        _, _, h, w = x_tensor.shape
        
        # Scegli una dimensione casuale per il resize (es. tra l'85% e il 100% dell'originale)
        resize_ratio = random.uniform(0.85, 1.0)
        new_h, new_w = int(h * resize_ratio), int(w * resize_ratio)
        
        # Ridimensiona l'immagine
        resized = F.interpolate(x_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # Calcola il padding casuale per riportare l'immagine alla dimensione originale (H, W)
        pad_top = random.randint(0, h - new_h)
        pad_bottom = (h - new_h) - pad_top
        pad_left = random.randint(0, w - new_w)
        pad_right = (w - new_w) - pad_left
        
        # Applica il padding (valore di riempimento a 0 o neutro)
        padded = F.pad(resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
        return padded

    def perturb(self, img_orig_tensor, epsilon, alpha=2.0, iterations=150):
        """
        Esegue il Transfer PGD potenziato con Momentum e Diverse Inputs,
        verificando l'evasione sul DSFD originale.
        """
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        success = False
        success_iteration = None

        # Se non è presente un modello surrogato, esegue il PGD standard
        if self.surrogate_net is None:
            return super().perturb(img_orig_tensor, epsilon=epsilon, alpha=alpha, iterations=iterations)

        # Inizializzazione del vettore di Momentum (MI-FGSM)
        g = torch.zeros_like(img_adv, device=self.device)

        for i in range(iterations):
            img_adv.requires_grad_(True)
            
            # 1. Applicazione Diverse Inputs (DI-FGSM) prima del forward sul surrogato
            x_in = self.diverse_inputs(img_adv)
            
            # Adatta l'input in base al modello surrogato (es. normalizzazione [0, 1] per YOLO)
            input_surrogate = x_in.unsqueeze(0) / 255.0
            
            surrogate_out = self.surrogate_net(input_surrogate)
            
            # 2. Calcolo Loss di evasione basata sull'output del surrogato
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

            # 3. Backward pass sul surrogato
            self.surrogate_net.zero_grad()
            if img_adv.grad is not None:
                img_adv.grad.zero_()
                
            loss.backward()

            if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
                break

            grad = img_adv.grad.detach()

            # 4. Integrazione Momentum (MI-FGSM)
            if self.use_momentum:
                # Normalizzazione L1 del gradiente per stabilire la velocità
                grad = grad / torch.mean(torch.abs(grad), dim=tuple(range(1, grad.dim())), keepdim=True)
                g = self.mu * g + grad
                update_dir = g.sign()
            else:
                update_dir = grad.sign()

            # 5. Aggiornamento PGD e Proiezione L-inf su scala [0, 255]
            with torch.no_grad():
                img_adv = img_adv - alpha * update_dir
                eta = img_adv - img_orig_tensor
                eta = torch.clamp(eta, min=-epsilon, max=epsilon)
                img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

            # 6. Check Evasione sul detector target reale (DSFD / DeepPrivacy2) ogni 5 iterazioni
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
