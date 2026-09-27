import torch
import torch.nn as nn
from tqdm import tqdm
from attacks.base_attack import BaseAttack

class UAPAttack(BaseAttack):
    """
    Universal Adversarial Perturbation (UAP) Attack contro il rilevatore facciale DSFD.
    Ricalcola la perturbazione in base all'epsilon fornito.
    """
    def __init__(self, detector_wrapper, dsfd_net, mean_tensor, device="cuda"):
        super().__init__(detector=detector_wrapper, device=device)
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net
        self.mean_tensor = mean_tensor
        self.uap_perturbation = None

    def fit(self, dataloader, epsilon=16.0, alpha=1.0, epochs=5, max_iter_per_img=15):
        """
        Calcola la perturbazione universale addestrandola specificamente per il budget epsilon indicato.
        """
        self.dsfd_net.eval()
        
        # Inizializziamo una nuova UAP a zero per questo specifico epsilon
        sample_batch = next(iter(dataloader))
        sample_img = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch
        c, h, w = sample_img.shape[1:] if sample_img.ndim == 4 else sample_img.shape
            
        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, dtype=torch.float32)

        print(f"[INFO] Avvio calcolo UAP dedicata per epsilon={epsilon} (Epoche: {epochs})")

        for epoch in range(epochs):
            fooled_count = 0
            total_images = 0

            for batch in dataloader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch

                for img in images:
                    total_images += 1
                    img_tensor = img.clone().detach().to(self.device).float()
                    if img_tensor.ndim == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    img_tensor = img_tensor.squeeze(0)

                    # Verifica se il volto è già evaso con la UAP corrente
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = current_adv.byte().float()
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1
                            continue

                    # Step locali di gradient ascent
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

                        loss = sum(loss_terms) if loss_terms else x_adv.sum()

                        self.dsfd_net.zero_grad()
                        if x_adv.grad is not None:
                            x_adv.grad.zero_()
                        loss.backward()

                        if x_adv.grad is None:
                            break

                        grad_sign = x_adv.grad.sign().squeeze(0)

                        with torch.no_grad():
                            # Aggiornamento e clamping stretto al budget epsilon corrente
                            self.uap_perturbation = self.uap_perturbation - alpha * grad_sign
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
            print(f"[UAP e={epsilon} | Epoca {epoch+1}/{epochs}] Fooling Rate Training: {fooling_rate:.2f}%")

        print(f"[INFO] UAP completata per epsilon={epsilon}")

    def perturb(self, img_orig_tensor, **kwargs):
        """
        Applica la UAP pre-calcolata per l'epsilon in corso.
        """
        if self.uap_perturbation is None:
            raise ValueError("La UAP non è stata calcolata! Esegui prima .fit(dataloader, epsilon).")

        x = img_orig_tensor.clone().detach().to(self.device).float()
        img_adv = torch.clamp(x + self.uap_perturbation, 0.0, 255.0)

        with torch.no_grad():
            eval_input = img_adv.detach().byte().float()
            if eval_input.ndim == 4:
                eval_input = eval_input.squeeze(0)
            dets = self.detector_wrapper(eval_input)
            success = (dets is None or len(dets) == 0 or len(dets[0]) == 0)

        return img_adv, success, 1

    def attack(self, img_orig_tensor, **kwargs):
        return self.perturb(img_orig_tensor, **kwargs)
