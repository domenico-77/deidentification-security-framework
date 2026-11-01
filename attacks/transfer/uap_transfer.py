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
        self.device = torch.device(device) if isinstance(device, str) else device
        self.detector_wrapper = detector_wrapper
        self.dsfd_net = dsfd_net.to(self.device) if dsfd_net is not None else None
        self.mean_tensor = mean_tensor.to(self.device) if mean_tensor is not None else None

        self.surrogate_net = surrogate_net.to(self.device) if surrogate_net is not None else None
        self.uap_perturbation = None

        try:
            super().__init__(detector_wrapper=detector_wrapper, dsfd_net=dsfd_net,
                             mean_tensor=mean_tensor, device=device)
        except Exception:
            pass

        if surrogate_net is not None:
            self.surrogate_net = surrogate_net.to(self.device)

    @staticmethod
    def _count_faces(dets) -> int:
        """Numero di volti rilevati, gestendo in modo sicuro i ritorni del
        DlibDetectorWrapper: None, [], [None], [tensor(N, 5)]."""
        if dets is None:
            return 0
        try:
            n = len(dets)
        except TypeError:
            return 0
        if n == 0:
            return 0
        first = dets[0]
        if first is None:
            return 0
        try:
            return len(first)
        except TypeError:
            return 0

    def fit(self, dataloader, epsilon, alpha=2, epochs=8, max_iter_per_img=15):
        if self.surrogate_net is None:
            print("[WARNING] Surrogato non rilevato, esecuzione fit standard su DSFD.")
            return super().fit(dataloader, epsilon=epsilon, alpha=alpha,
                               epochs=epochs, max_iter_per_img=max_iter_per_img)
        else:
            print(f"[SUCCESSO] Modello surrogato attivo in TransferUAPAttack. "
                  f"Avvio calcolo UAP (epsilon={epsilon})...")

        if hasattr(self.surrogate_net, "eval") and callable(self.surrogate_net.eval):
            try:
                self.surrogate_net.eval()
            except Exception:
                pass
        elif hasattr(self.surrogate_net, "net") and hasattr(self.surrogate_net.net, "eval"):
            try:
                self.surrogate_net.net.eval()
            except Exception:
                pass

        sample_batch = next(iter(dataloader))
        sample_img = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch

        if sample_img.ndim == 4:
            _, c, h, w = sample_img.shape
        else:
            c, h, w = sample_img.shape[0], sample_img.shape[1], sample_img.shape[2]

        self.uap_perturbation = torch.zeros((c, h, w), device=self.device, dtype=torch.float32)

        print(f"[INFO] Inizio calcolo UAP Transfer guidato dal surrogato: "
              f"epsilon={epsilon}, alpha={alpha}, epochs={epochs}")

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

                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = (current_adv.squeeze(0).byte().float()
                                      if current_adv.ndim == 4
                                      else current_adv.byte().float())
                        if eval_input.ndim == 4:
                            eval_input = eval_input[0]
                        dets = self.detector_wrapper(eval_input)

                        if self._count_faces(dets) == 0:
                            fooled_count += 1
                            continue

                    delta_local = self.uap_perturbation.clone().detach().requires_grad_(True)

                    for _ in range(max_iter_per_img):
                        delta_local.requires_grad_(True)
                        x_adv = torch.clamp(img_tensor + delta_local, 0.0, 255.0)

                        input_surrogate = x_adv / 255.0 if x_adv.max() > 1.0 else x_adv
                        if input_surrogate.ndim == 3:
                            input_surrogate = input_surrogate.unsqueeze(0)

                        if hasattr(self.surrogate_net, "compute_adversarial_loss"):
                            loss = self.surrogate_net.compute_adversarial_loss(input_surrogate)
                        else:
                            try:
                                surrogate_out = self.surrogate_net(input_surrogate)
                                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                                if isinstance(surrogate_out, (list, tuple)):
                                    for item in surrogate_out:
                                        if isinstance(item, torch.Tensor) and item.numel() > 1:
                                            loss = loss + item.abs().sum()
                                elif isinstance(surrogate_out, torch.Tensor):
                                    loss = surrogate_out.abs().sum()
                            except Exception:
                                loss = (x_adv - img_tensor).norm()

                        if not loss.requires_grad:
                            loss = loss.clone().requires_grad_(True)

                        self.surrogate_net.zero_grad()
                        if delta_local.grad is not None:
                            delta_local.grad.zero_()

                        loss.backward()

                        if delta_local.grad is not None:
                            with torch.no_grad():
                                grad_sign = delta_local.grad.sign()
                                delta_local = delta_local + alpha * grad_sign
                                delta_local = torch.clamp(delta_local, min=-epsilon, max=epsilon)
                                delta_local = delta_local.detach()
                        else:
                            break

                    with torch.no_grad():
                        self.uap_perturbation = 0.85 * self.uap_perturbation + 0.15 * delta_local
                        self.uap_perturbation = torch.clamp(self.uap_perturbation,
                                                            min=-epsilon, max=epsilon)

                    with torch.no_grad():
                        current_adv = torch.clamp(img_tensor + self.uap_perturbation, 0.0, 255.0)
                        eval_input = (current_adv.squeeze(0).byte().float()
                                      if current_adv.ndim == 4
                                      else current_adv.byte().float())
                        if eval_input.ndim == 4:
                            eval_input = eval_input[0]
                        dets = self.detector_wrapper(eval_input)

                        if self._count_faces(dets) == 0:
                            fooled_count += 1

            fooling_rate = (fooled_count / total_images) * 100 if total_images > 0 else 0.0
            print(f"[UAP Transfer Epoch {epoch+1}/{epochs}] "
                  f"Fooling Rate sul dataset: {fooling_rate:.2f}%")

        print("[INFO] Calcolo UAP sul surrogato completato con successo!")

    def perturb(self, img_orig_tensor, epsilon, **kwargs):
        """Applica la UAP ottimizzata e verifica l'evasione sul detector target.

        Override esplicito per gestire correttamente i ritorni del DlibDetectorWrapper
        (che usa `[None]` per "nessun volto rilevato" invece di `[]`).
        """
        if self.uap_perturbation is None:
            # Nessun fit eseguito: fallback a UAP standard
            return super().perturb(img_orig_tensor, epsilon, **kwargs)

        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)

        # Applica la perturbazione universale (broadcast su H, W)
        with torch.no_grad():
            img_adv = torch.clamp(img_adv + self.uap_perturbation, 0.0, 255.0)

        # Verifica evasione
        detector_input = img_adv.detach().byte().float()
        if detector_input.ndim == 4:
            detector_input = detector_input[0]

        with torch.no_grad():
            dets = self.detector_wrapper(detector_input)

        # FIX: stesso helper di _count_faces, niente len(dets[0]) su None
        success = (self._count_faces(dets) == 0)

        return img_adv, success, 1
