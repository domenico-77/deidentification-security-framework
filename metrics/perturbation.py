import torch

def calculate_perturbation_metrics(orig_tensor, adv_tensor):
    """
    Calcola le metriche di perturbazione (es. L2, Linf) tra l'immagine originale
    e quella avversaria, garantendo la compatibilità dei dispositivi.
    """
    # Sposta entrambi i tensori sullo stesso dispositivo (preferibilmente CPU per i calcoli metrici o GPU)
    device = orig_tensor.device
    adv_tensor = adv_tensor.to(device)
    
    orig_f = orig_tensor.float()
    adv_f = adv_tensor.float()
    
    # Calcolo delle differenze
    diff = adv_f - orig_f
    
    # Calcolo delle norme di perturbazione
    l2_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1).mean().item()
    linf_norm = torch.max(torch.abs(diff.view(diff.size(0), -1)), dim=1)[0].mean().item()
    
    return {
        "l2": l2_norm,
        "linf": linf_norm
    }
