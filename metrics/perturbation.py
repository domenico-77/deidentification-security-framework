import torch


def calculate_perturbation_metrics(orig_tensor: torch.Tensor, adv_tensor: torch.Tensor):
    """Calcola le distanze di perturbazione L_inf, L2, MSE e PSNR."""
    diff = adv_tensor.float() - orig_tensor.float()

    l_inf = torch.max(torch.abs(diff)).item()
    l2 = torch.norm(diff).item()
    mse = torch.mean(diff ** 2).item()

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * torch.log10(255.0 / torch.sqrt(torch.tensor(mse))).item()

    return {
        "l_inf": l_inf,
        "l_2": l2,
        "mse": mse,
        "psnr_db": psnr
    }