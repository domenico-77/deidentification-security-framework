import torch
import torch.nn as nn
from tqdm import tqdm
from attacks.base_attack import BaseAttack

class UAPAttack(BaseAttack):
    """
    Universal Adversarial Perturbation (UAP) Attack contro il rilevatore facciale DSFD.
    Eredita da BaseAttack per mantenere coerenza con il framework di sicurezza IA.
    """
    def __init__(self, detector_wrapper, dsfd_net, mean_tensor, device="cuda"):
        # Passiamo correttamente il detector alla classe base BaseAttack
        super().__init__(detector=detector_wrapper, device=device)
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net
        self.mean_tensor = mean_tensor
        self.uap_perturbation = None  # Verrà inizializzata durante il .fit()

    def fit(self, dataloader, epsilon=16.0, alpha=1.0, epochs=5, max_iter_per_img=15):
        """
        Calcola la perturbazione universale iterando sul dataloader di training.
        """
        self.dsfd_net.eval()
        
        # Inizializziamo la perturbazione universale a zero basandoci sul primo batch
        sample_batch = next(iter(dataloader))
        if isinstance(sample_batch, (list, tuple)):
            sample_img = sample_batch[0]
        else:
            sample_img = sample_batch

        if sample_img.ndim == 4:
            _, c, h, w = sample_img.shape
        else:
            c, h, w = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]
            
        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, dtype=torch.float32)

        print(f"[INFO] Inizio calcolo UAP: epsilon={epsilon}, alpha={alpha}, epochs={epochs}")

        for epoch in range(epochs):
            fooled_count = 0
            total_images = 0

            for batch in tqdm(dataloader, desc=f"[UAP Epoch {epoch+1}/{epochs}]"):
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                for img in images:
                    total_images += 1
                    img_tensor = img.clone().detach().to(self.device).float()
                    
                    if img_tensor.ndim == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    
                    img_tensor = img_tensor.squeeze(0)

                    # Verifichiamo se con la UAP corrente il volto è già ingannato
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = current_adv.byte().float()
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1
                            continue

                    # Altrimenti, eseguiamo step di gradient ascent locali
                    x_adv = img_tensor + self.uap_perturbation.clone().detach()
                    x_adv = torch.clamp(x_adv, 0.0, 255.0)

                    for _ in range(max_iter_per_img):
                        x_adv.requires_grad_(True)
                        input_net = x_adv.unsqueeze(0) - self.mean_tensor
                        
                        net_out = self.dsfd_net(input_net, 0.0, 0.0)
                        
                        loss_terms = []
                        if isinstance(net_out, (list, tuple)):
                            for t in net_out:
                                if isinstance(t, torch.Tensor):
                                    if t.ndim >= 2 and t.shape[-1] == 2:
                                        loss_terms.append(torch.relu(t[..., 1] - t[..., 0]).sum())
                                    else:
                                        loss_terms.append(torch.relu(t).sum())
                        elif isinstance(net_out, torch.Tensor):
                            loss_terms.append(torch.relu(net_out).sum())

                        if len(loss_terms) > 0:
                            loss = sum(loss_terms)
                        else:
                            loss = x_adv.sum()

                        self.dsfd_net.zero_grad()
                        if x_adv.grad is not None:
                            x_adv.grad.zero_()
                            
                        loss.backward()

                        if x_adv.grad is None:
                            break

                        grad_sign = x_adv.grad.sign().squeeze(0)

                        with torch.no_grad():
                            self.uap_perturbation = self.uap_perturbation + alpha * grad_sign
                            self.uap_perturbation = torch.clamp(self.uap_perturbation, min=-epsilon, max=epsilon)

                        with torch.no_grad():
                            current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                            eval_input = current_adv.byte().float()
                            if eval_input.ndim == 4:
                                eval_input = eval_input.squeeze(0)
                            dets = self.detector_wrapper(eval_input)
                            if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                                fooled_count += 1
                                break

            fooling_rate = (fooled_count / total_images) * 100 if total_images > 0 else 0.0
            print(f"[UAP Epoch {epoch+1}/{epochs}] Fooling Rate sul dataset: {fooling_rate:.2f}%")

        print("[INFO] Calcolo UAP completato con successo!")

    def perturb(self, img_orig_tensor, **kwargs):
        """
        Applica la perturbazione universale pre-calcolata a un'immagine di test,
        rispettando l'epsilon eventualmente passato dai kwargs del benchmark runner.
        """
        if self.uap_perturbation is None:
            raise ValueError("La UAP non è stata calcolata! Esegui prima .fit(dataloader).")

        x = img_orig_tensor.clone().detach().to(self.device).float()
        
        epsilon_eval = kwargs.get("epsilon", None)
        
        if epsilon_eval is not None:
            uap_norm = torch.max(torch.abs(self.uap_perturbation))
            if uap_norm > 0:
                scale = min(1.0, epsilon_eval / uap_norm.item())
                current_uap = self.uap_perturbation * scale
            else:
                current_uap = self.uap_perturbation
        else:
            current_uap = self.uap_perturbation

        img_adv = torch.clamp(x + current_uap, 0.0, 255.0)

        with torch.no_grad():
            eval_input = img_adv.detach().byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input.squeeze(0)
                
            dets = self.detector_wrapper(eval_input)
            success = (dets is None or len(dets) == 0 or len(dets[0]) == 0)

        succ_iter = 1 
        return img_adv, success, succ_iter

    def attack(self, img_orig_tensor, **kwargs):
        """
        Implementazione del metodo astratto obbligatorio richiesto da BaseAttack.
        Delega l'esecuzione direttamente a perturb().
        """
        return self.perturb(img_orig_tensor, **kwargs)
