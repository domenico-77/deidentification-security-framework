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
        super().__init__(detector=detector_wrapper, device=device)
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net
        self.mean_tensor = mean_tensor
        self.uap_perturbation = None  # Verrà inizializzata durante il .fit()

    def fit(self, dataloader, epsilon=16.0, alpha=2.0, epochs=5, max_iter_per_img=15):
        """
        Calcola la perturbazione universale iterando sul dataloader di training.
        """
        self.dsfd_net.eval()
        
        # Inizializziamo la perturbazione universale a zero basandoci sul primo batch
        sample_batch = next(iter(dataloader))
        sample_img = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch

        if sample_img.ndim == 4:
            _, c, h, w = sample_img.shape
        else:
            c, h, w = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]
            
        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, dtype=torch.float32)

        print(f"[INFO] Inizio calcolo UAP globale: epsilon={epsilon}, alpha={alpha}, epochs={epochs}")

        for epoch in range(epochs):
            fooled_count = 0
            total_images = 0

            for batch in tqdm(dataloader, desc=f"[UAP Epoch {epoch+1}/{epochs}]"):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device).float()

                if images.ndim == 3:
                    images = images.unsqueeze(0)

                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    total_images += 1

                    # 1. Verifica se con la UAP corrente il volto è già ingannato
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        # Garantiamo il formato [C, H, W] richiesto dal detector
                        eval_input = current_adv.squeeze(0).byte().float() if current_adv.ndim == 4 else current_adv.byte().float()
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1
                            continue

                    # 2. Ottimizzazione locale per spingere il detector a perdere il volto
                    delta_local = self.uap_perturbation.clone().detach().requires_grad_(True)

                    for _ in range(max_iter_per_img):
                        delta_local.requires_grad_(True)
                        x_adv = torch.clamp(img_tensor + delta_local, 0.0, 255.0)
                        input_net = x_adv.unsqueeze(0) - self.mean_tensor
                        
                        net_out = self.dsfd_net(input_net, 0.0, 0.0)
                        
                        # Estraiamo le confidenze [1, 5333, 5] -> canale 4
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        if isinstance(net_out, torch.Tensor) and net_out.ndim == 3 and net_out.shape[-1] == 5:
                            confidences = net_out[..., 4]
                            loss = confidences.mean()
                        elif isinstance(net_out, torch.Tensor):
                            loss = net_out.mean()
                        elif isinstance(net_out, (list, tuple)):
                            loss = sum([t.mean() for t in net_out if isinstance(t, torch.Tensor)])

                        self.dsfd_net.zero_grad()
                        if delta_local.grad is not None:
                            delta_local.grad.zero_()
                            
                        loss.backward()

                        if delta_local.grad is not None:
                            with torch.no_grad():
                                grad_sign = delta_local.grad.sign()
                                delta_local = delta_local - alpha * grad_sign
                                delta_local = torch.clamp(delta_local, min=-epsilon, max=epsilon)
                                delta_local = delta_local.detach()
                        else:
                            break

                    # 3. Aggiornamento globale stabile della UAP (EMA)
                    with torch.no_grad():
                        self.uap_perturbation = 0.85 * self.uap_perturbation + 0.15 * delta_local
                        self.uap_perturbation = torch.clamp(self.uap_perturbation, min=-epsilon, max=epsilon)

                    # 4. Controllo post-aggiornamento
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = current_adv.squeeze(0).byte().float() if current_adv.ndim == 4 else current_adv.byte().float()
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1

            fooling_rate = (fooled_count / total_images) * 100 if total_images > 0 else 0.0
            print(f"[UAP Epoch {epoch+1}/{epochs}] Fooling Rate sul dataset: {fooling_rate:.2f}%")

        print("[INFO] Calcolo UAP completato con successo!")

    def perturb(self, img_orig_tensor, **kwargs):
        """
        Applica la perturbazione universale pre-calcolata a un'immagine di test.
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
            eval_input = img_adv.squeeze(0).byte().float() if img_adv.ndim == 4 else img_adv.byte().float()
            dets = self.detector_wrapper(eval_input)
            success = (dets is None or len(dets) == 0 or len(dets[0]) == 0)

        return img_adv, success, 1

    def attack(self, img_orig_tensor, **kwargs):
        """
        Implementazione del metodo astratto richiesto da BaseAttack.
        """
        return self.perturb(img_orig_tensor, **kwargs)
