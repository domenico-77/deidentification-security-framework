import torch
from attacks.base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method (FGSM) Single-step attack."""

    def attack(self, image_tensor: torch.Tensor, epsilon: float = 32.0) -> torch.Tensor:
        if image_tensor.dim() == 3:
            orig_tensor = image_tensor.unsqueeze(0).to(self.device).float().clone()
        else:
            orig_tensor = image_tensor.to(self.device).float().clone()

        orig_tensor.requires_grad = True
        loss = self.detector.compute_adversarial_loss(orig_tensor)

        if loss is not None:
            loss.backward()
            grad_sign = orig_tensor.grad.sign()
            adv_tensor = orig_tensor - epsilon * grad_sign
            adv_tensor = torch.clamp(adv_tensor, 0.0, 255.0)
            return adv_tensor.detach().squeeze(0)

        return orig_tensor.detach().squeeze(0)