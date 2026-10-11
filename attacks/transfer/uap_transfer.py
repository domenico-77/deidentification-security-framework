import torch
from tqdm import tqdm
from attacks.uap import UAPAttack

class TransferUAPAttack(UAPAttack):
    """
    Versione estesa di UAP per attacchi di Transfer / Gray-Box,
    sfruttando un modello surrogato (YOLO o RetinaFace) per l'ottimizzazione
    della perturbazione universale e verificando l'evasione sul target (DSFD).
    """
    def __init__(self, detector_wrapper, dsfd_net, mean_tensor, surrogate_net=None, device="cuda"):
        super().__init__(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net, mean_tensor=mean_tensor, device=device)
        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None

    def fit(self, dataloader, epsilon=16.0, alpha=2, epochs=8, max_iter_per_img=15):
        """
        Calcola la perturbazione universale (UAP) guidata dal modello surrogato.
        Se il surrogato non è disponibile, esegue il fit standard basato su DSFD.
        """
        if self.surrogate_net is None:
            print("[INFO] Nessun modello surrogato fornito per UAP. Esecuzione fit standard su DSFD.")
            return super().fit(dataloader, epsilon=epsilon, alpha=alpha, epochs=epochs, max_iter_per_img=max_iter_per_img)

        self.surrogate_net.eval()
        
        sample_batch = next(iter(dataloader))
        sample_img = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch

        if sample_img.ndim == 4:
            _, c, h, w = sample_img.shape
        else:
            c, h, w = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]
            
        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, dtype=torch.float32)

        print(f"[INFO] Inizio calcolo UAP Transfer guidato dal surrogato: epsilon={epsilon}, alpha={alpha}, epochs={epochs}")

        for epoch in range(epochs):
            fooled_count = 0
            total_images = 0

            for batch in tqdm(dataloader, desc=f"[UAP Transfer Epoch {epoch+1}/{epochs}]"):
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(self.device).float()

                if images.ndim == 3:
                    images = images.unsqueeze(0)

                for i in range(images.shape[0]):
                    img_tensor = images[i]
                    total_images += 1

                    # 1. Check rapido con la UAP corrente sul detector target (DSFD)
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = current_adv.squeeze(0).byte().float() if current_adv.ndim == 4 else current_adv.byte().float()
                        if eval_input.ndim == 4:
                            eval_input = eval_input[0]
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1
                            continue

                    # 2. Ottimizzazione locale guidata dal modello surrogato
                    delta_local = self.uap_perturbation.clone().detach().requires_grad_(True)

                    for _ in range(max_iter_per_img):
                        delta_local.requires_grad_(True)
                        x_adv = torch.clamp(img_tensor + delta_local, 0.0, 255.0)
                        
                        # Surrogato richiede 4D [B, C, H, W] in [0, 1]
                        input_surrogate = x_adv / 255.0 if x_adv.max() > 1.0 else x_adv
                        if input_surrogate.ndim == 3:
                            input_surrogate = input_surrogate.unsqueeze(0)

                        if hasattr(self.surrogate_net, "compute_adversarial_loss"):
                            loss = self.surrogate_net.compute_adversarial_loss(input_surrogate)
                        else:
                            surrogate_out = self.surrogate_net(input_surrogate)
                            loss = torch.relu(surrogate_out[1]).sum() if isinstance(surrogate_out, (list, tuple)) else torch.tensor(0.0, device=self.device, requires_grad=True)

                        self.surrogate_net.zero_grad()
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

                    # 3. Aggiornamento globale EMA
                    with torch.no_grad():
                        self.uap_perturbation = 0.80 * self.uap_perturbation + 0.20 * delta_local
                        self.uap_perturbation = torch.clamp(self.uap_perturbation, min=-epsilon, max=epsilon)

                    # 4. Controllo post-aggiornamento su DSFD
                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = current_adv.squeeze(0).byte().float() if current_adv.ndim == 4 else current_adv.byte().float()
                        if eval_input.ndim == 4:
                            eval_input = eval_input[0]
                        dets = self.detector_wrapper(eval_input)
                        if dets is None or len(dets) == 0 or len(dets[0]) == 0:
                            fooled_count += 1

            fooling_rate = (fooled_count / total_images) * 100 if total_images > 0 else 0.0
            print(f"[UAP Transfer Epoch {epoch+1}/{epochs}] Fooling Rate sul target DSFD: {fooling_rate:.2f}%")

        print("[INFO] Calcolo UAP Transfer completato con successo!")
