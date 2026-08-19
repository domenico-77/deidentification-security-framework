import numpy as np

def compute_pipeline_mse(orig_img: np.ndarray, anon_img: np.ndarray) -> float:
    """Calcola l'MSE tra l'input e l'output della pipeline. Prossimo a 0 se l'anonimizzazione viene saltata."""
    return float(np.mean((orig_img.astype(float) - anon_img.astype(float)) ** 2))