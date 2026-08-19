import torch
from attacks.pgd import PGDAttack


class BIMAttack(PGDAttack):
    """Basic Iterative Method (BIM) - PGD senza inizializzazione casuale."""

    def attack(self, image_tensor: torch.Tensor, epsilon: float = 32.0, alpha: float = 2.0,
               steps: int = 50) -> torch.Tensor:
        # BIM equivale a PGD senza random restart
        return super().attack(image_tensor, epsilon=epsilon, alpha=alpha, steps=steps)