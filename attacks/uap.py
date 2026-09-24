import torch
import torch.nn as nn
import numpy as np
from attacks.base_attack import BaseAttack

class UAPAttack(BaseAttack):
    def __init__(self, detector=None, detector_wrapper=None, dsfd_net=None, mean_tensor=None, device: torch.device = None):
        det = detector if detector is not None else detector_wrapper
        super().__init__(detector=det, device=device)
        
        self.detector_wrapper = self.detector
        self.dsfd_net = dsfd_net.to(self.device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(self.device) if mean_tensor is not None else None
        
        # Perturbazione universale (inizializzata a None, verrà popolata con fit() o caricata)
        self.uap_perturbation = None

    def fit(self, dataloader, epsilon=16.0, alpha=2.0, epochs=3, max_iter_per_img=10):
        """
        Calcola la perturbazione universale (UAP) iterando su un dataset di immagini.
        """
        print("[INFO] Avvio calcolo della Universal Adversarial Perturbation (UAP)...")
        
        # 1. Otteniamo la forma di un'immagine campione dal dataloader per inizializzare v
        sample_batch = next(iter(dataloader))
        sample_img = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        
        if sample_img.ndim == 4:
            sample_img = sample_img[0]
            
        c, h, w = sample_img.shape
        # Inizializziamo la perturbazione universale v a zero sul device corretto
        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, requires_grad=False)

        fooling_rate = 0.0
        total_images = 0

        for epoch in range(epochs):
            fooled_count = 0
            total_images = 0

            for batch in dataloader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                for img in images:
                    total_images += 1
                    img_tensor = img.clone().detach().to(self.device).float()
                    
                    if img_tensor.ndim == 3 and img_tensor.shape[0] != c:
                        # Se per caso è [H, W, C], lo portiamo a [C, H, W]
                        img_tensor = img_tensor.permute(2, 0, 1)

                    # Applichiamo la UAP corrente
                    x_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0).clone().detach()
                    x_adv.requires_grad_(True)

                    # Verifichiamo se il volto è ancora rilevato (assicurandoci che sia 3D [C, H, W])
                    with torch.no_grad():
                        eval_input = x_adv.detach().byte().float()
                        if eval_input.ndim == 4:
                            eval_input = eval_input.squeeze(0)
                        dets = self.detector_wrapper(eval_input)
                        face_detected = (dets is not None and len(dets) > 0 and len(dets[0]) > 0)

                    if not face_detected:
                        fooled_count += 1
                        continue # Se è già evaso, passiamo all'immagine successiva

                    # Altrimenti, eseguiamo qualche step di gradient ascent
                    for _ in range(max_iter_per_img):
                        x_adv.requires_grad_(True)
                        input_net = x_adv.unsqueeze(0) - self.mean_tensor
                        
                        net_out = self.dsfd_net(input_net, 0.0, 0.0)
                        
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        if isinstance(net_out, (list, tuple)):
                            for t in net_out:
                                if isinstance(t, torch.Tensor):
                                    if t.ndim >= 2 and t.shape[-1] == 2:
                                        loss = loss + torch.relu(t[..., 1] - t[..., 0]).sum()
                                    else:
                                        loss = loss + torch.relu(t).sum()
                        elif isinstance(net_out, torch.Tensor):
                            loss = loss + torch.relu(net_out).sum()

                        self.dsfd_net.zero_grad()
                        loss.backward()

                        if x_adv.grad is None:
                            break

                        grad_sign = x_adv.grad.sign().squeeze(0)

                        with torch.no_grad():
                            # Aggiorniamo la perturbazione universale v accumulando la direzione
                            self.uap_perturbation = self.uap_perturbation - alpha * grad_sign
                            # Proiezione sulla palla L-infinito di raggio epsilon
                            self.uap_perturbation = torch.clamp(self.uap_perturbation, min=-epsilon, max=epsilon)

                        # Ricontrolliamo l'evasione (formato 3D [C, H, W])
                        with torch.no_grad():
                            current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                            eval_input = current_adv.byte().float()
                            if eval_input.ndim == 4:
                                eval_input = eval_input.squeeze(0)
                            dets = self.detector_wrapper(eval_input)
                            if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                                break

            fooling_rate = (fooled_count / max(1, total_images)) * 100
            print(f"[UAP Epoch {epoch+1}/{epochs}] Fooling Rate sul dataset: {fooling_rate:.2f}%")

        print("[INFO] Calcolo UAP completato con successo!")

    def perturb(self, img_orig_tensor, **kwargs):
        """
        Applica la perturbazione universale pre-calcolata a un'immagine di test,
        rispettando l'epsilon eventualmente passato dai kwargs del benchmark runner.
        """
        if self.uap_perturbation is None:
            raise ValueError("La UAP non è stata calcolata! Esegui prima .fit(dataloader) o carica un tensore salvato.")

        x = img_orig_tensor.clone().detach().to(self.device).float()
        
        # Recupera l'epsilon dinamico dal runner se presente, altrimenti usa la norma della UAP o un default
        epsilon_eval = kwargs.get("epsilon", None)
        
        if epsilon_eval is not None:
            # Scala o limita la UAP fissa in base all'epsilon del test corrente (mantenendo la direzione)
            uap_norm = torch.max(torch.abs(self.uap_perturbation))
            if uap_norm > 0:
                scale = min(1.0, epsilon_eval / uap_norm.item())
                current_uap = self.uap_perturbation * scale
            else:
                current_uap = self.uap_perturbation
        else:
            current_uap = self.uap_perturbation

        # Applicazione della UAP vincolata nello spazio pixel [0, 255]
        img_adv = torch.clamp(x + current_uap, 0.0, 255.0)

        # Valutazione del successo dell'attacco sull'immagine singola (formato 3D [C, H, W])
        with torch.no_grad():
            eval_input = img_adv.detach().byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input.squeeze(0)
                
            dets = self.detector_wrapper(eval_input)
            success = (dets is None or len(dets) == 0 or len(dets[0]) == 0)

        succ_iter = 1 
        return img_adv, success, succ_iter

    def attack(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Metodo astratto obbligatorio richiesto da BaseAttack.
        Restituisce direttamente il tensore adversarial perturbato.
        """
        img_adv, _, _ = self.perturb(image_tensor, **kwargs)
        return img_adv
