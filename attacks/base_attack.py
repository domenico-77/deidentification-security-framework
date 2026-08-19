from abc import ABC, abstractmethod
import torch


class BaseAttack(ABC):
    """Interfaccia astratta per tutti gli attacchi adversarial."""

    def __init__(self, detector, device: torch.device = None):
        self.detector = detector
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def attack(self, image_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Esegue l'attacco su un tensore immagine [C, H, W] o [B, C, H, W] in scala [0, 255].
        Restituisce il tensore adversarial perturbato.
        """
        pass