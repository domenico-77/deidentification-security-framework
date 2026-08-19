import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack


class PGDAttack(BaseAttack):
    def __init__(self, detector, device=None):
        super().__init__(detector, device)

    def attack(self, image_tensor: torch.Tensor, epsilon: float = 32.0, alpha: float = 2.0,
               steps: int = 150) -> torch.Tensor:
        """
        Attacco PGD $L_\infty$ mirato a sopprimere le detection facciali.
        """
        self.detector.model.eval()

        if image_tensor.dim() == 3:
            orig_tensor = image_tensor.unsqueeze(0).to(self.device).float()
        else:
            orig_tensor = image_tensor.to(self.device).float()

        # Inizializzazione casuale all'interno della pallina epsilon
        delta = torch.zeros_like(orig_tensor, requires_grad=True).to(self.device)
        delta.data.uniform_(-epsilon, epsilon)
        delta.data = torch.clamp(orig_tensor + delta.data, 0.0, 255.0) - orig_tensor

        for step in range(steps):
            delta.requires_grad_(True)
            adv_tensor = orig_tensor + delta

            # Forward pass nel detector
            loss = self.detector.compute_adversarial_loss(adv_tensor)

            if loss is None:  # Nessuna detection, early stopping
                break

            loss.backward()

            with torch.no_grad():
                grad_sign = delta.grad.sign()
                delta.data = delta.data - alpha * grad_sign
                # Proiezione L_infinity
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                # Clamp nel range valido dei pixel [0, 255]
                delta.data = torch.clamp(orig_tensor + delta.data, 0.0, 255.0) - orig_tensor

            delta.grad.zero_()

            # Early stopping dinamico se il detector non rileva più volti
            with torch.no_grad():
                curr_adv = torch.clamp(orig_tensor + delta, 0.0, 255.0)
                if self.detector.count_detections(curr_adv) == 0:
                    break

        return torch.clamp(orig_tensor + delta, 0.0, 255.0).detach().squeeze(0)