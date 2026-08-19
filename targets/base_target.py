from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseDeidentificationTarget(ABC):

    @abstractmethod
    def process_image(self, image: torch.Tensor) -> np.ndarray:
        """Processa l'immagine attraverso la pipeline di anonimizzazione completa."""
        pass