import torch
from attacks.base_attack import BaseAttack


class UniversalAdversarialPerturbation(BaseAttack):
    """Universal Adversarial Perturbation (UAP) su un insieme di immagini."""

    def generate_uap(self, dataloader, epsilon: float = 16.0, max_iter: int = 10) -> torch.Tensor:
        # Inizializza perturbazione universale vuota
        sample_img, _ = next(iter(dataloader))
        v = torch.zeros_like(sample_img[0]).to(self.device)

        for epoch in range(max_iter):
            for imgs, _ in dataloader:
                for img in imgs:
                    img_adv = torch.clamp(img.to(self.device) + v, 0.0, 255.0)
                    if self.detector.count_detections(img_adv) > 0:
                        # Calcola delta individuale e aggiorna v
                        pass
        return v

    def attack(self, image_tensor: torch.Tensor, uap_mask: torch.Tensor) -> torch.Tensor:
        return torch.clamp(image_tensor + uap_mask, 0.0, 255.0)