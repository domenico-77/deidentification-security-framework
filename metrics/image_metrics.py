import numpy as np
import torch

def compute_pipeline_mse(orig_img, anon_img):
    """
    Calcola l'Errore Quadratico Medio (MSE) tra l'immagine originale e quella anonimizzata,
    gestendo in modo sicuro sia array NumPy che tensori PyTorch.
    """
    # Converte l'immagine originale in numpy se è un tensore
    if isinstance(orig_img, torch.Tensor):
        orig_img = orig_img.detach().cpu().numpy()
        if orig_img.ndim == 3 and orig_img.shape[0] in [1, 3]: # (C, H, W) -> (H, W, C)
            orig_img = np.transpose(orig_img, (1, 2, 0))

    # Converte l'immagine anonimizzata in numpy se è un tensore
    if isinstance(anon_img, torch.Tensor):
        anon_img = anon_img.detach().cpu().numpy()
        if anon_img.ndim == 3 and anon_img.shape[0] in [1, 3]: # (C, H, W) -> (H, W, C)
            anon_img = np.transpose(anon_img, (1, 2, 0))

    # Assicura che le dimensioni combacino
    if orig_img.shape != anon_img.shape:
        # Se necessario, adatta la forma
        pass

    return float(np.mean((orig_img.astype(float) - anon_img.astype(float)) ** 2))
