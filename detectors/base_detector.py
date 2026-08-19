from abc import ABC, abstractmethod
import torch


class BaseDetector(ABC):

    @abstractmethod
    def count_detections(self, image_tensor: torch.Tensor) -> int:
        """Restituisce il numero di volti rilevati."""
        pass

    @abstractmethod
    def compute_adversarial_loss(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Calcola la loss da minimizzare/massimizzare per ingannare il detector."""
        pass