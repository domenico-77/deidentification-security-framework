import torch
import numpy as np

def calculate_perturbation_metrics(orig_tensor, adv_tensor):
    """
    Calcola le metriche di perturbazione garantendo la compatibilità 
    dei dispositivi e la scala corretta [0, 255].
    """
    device = orig_tensor.device
    adv_tensor = adv_tensor.to(device)
    
    orig_f = orig_tensor.float()
    adv_f = adv_tensor.float()
    
    perturbation = adv_f - orig_f
    
    # Metriche
    linf = torch.max(torch.abs(perturbation)).item()
    l2 = torch.norm(perturbation, p=2).item()
    mse = torch.mean(perturbation ** 2).item()
    
    if mse > 0:
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    else:
        psnr = float("inf")
        
    return {
        "linf": linf,
        "l2": l2,
        "mse": mse,
        "psnr": psnr
    }
