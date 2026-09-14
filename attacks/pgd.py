import torch
import numpy as np

def run_pgd_attack(
    img_orig_tensor, 
    anonymizer, 
    mean_tensor, 
    device, 
    epsilon_max=32.0, 
    alpha=2.0, 
    iterations=150
):
    """
    Esegue l'attacco PGD white-box contro il detector DSFD di DeepPrivacy2.
    """
    detector_wrapper = anonymizer.detector
    dsfd_net = detector_wrapper.face_detector.net.to(device)
    dsfd_net.eval()

    # Funzione interna per il conteggio dei volti
    def count_faces(img_tensor):
        detector_input = img_tensor.detach().byte().float()
        with torch.no_grad():
            detections = detector_wrapper(detector_input)
        if len(detections) > 0 and detections[0] is not None:
            return len(detections[0])
        return 0

    clean_faces = count_faces(img_orig_tensor)
    
    # Inizializzazione immagine adversarial
    img_adv = img_orig_tensor.clone().detach()
    success = False
    success_iteration = None

    for i in range(iterations):
        img_adv.requires_grad_(True)
        
        # Forward pass sul DSFD
        input_batch = img_adv.unsqueeze(0)
        input_net = input_batch - mean_tensor
        net_out = dsfd_net(input_net, 0.0, 0.0)

        # Calcolo Loss
        loss = 0.0
        if isinstance(net_out, (list, tuple)):
            for t in net_out:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    if t.ndim >= 2 and t.shape[-1] == 2:
                        face_logits = t[..., 1]
                        bg_logits = t[..., 0]
                        loss = loss + torch.relu(face_logits - bg_logits).sum()
                    else:
                        loss = loss + torch.relu(t).sum()
        elif isinstance(net_out, torch.Tensor) and net_out.requires_grad:
            loss = torch.relu(net_out).sum()

        dsfd_net.zero_grad()
        if hasattr(anonymizer, "zero_grad"):
            anonymizer.zero_grad()

        loss.backward()

        if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
            break

        grad_sign = img_adv.grad.sign()

        # Aggiornamento PGD + Proiezione L-inf sul range [0, 255]
        with torch.no_grad():
            img_adv = img_adv - alpha * grad_sign
            eta = img_adv - img_orig_tensor
            eta = torch.clamp(eta, min=-epsilon_max, max=epsilon_max)
            img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

        # Early stopping ogni 5 iterazioni
        if i % 5 == 0 or i == iterations - 1:
            chk_faces = count_faces(img_adv)
            if chk_faces == 0:
                success = True
                success_iteration = i
                break

    # Metriche finali di perturbazione
    adv_float = img_adv.detach().float()
    orig_float = img_orig_tensor.detach().float()
    perturbation = adv_float - orig_float

    linf_real = torch.max(torch.abs(perturbation)).item()
    l2_real = torch.norm(perturbation, p=2).item()
    mse_pert = torch.mean(perturbation ** 2).item()
    psnr = 10.0 * np.log10((255.0 ** 2) / mse_pert) if mse_pert > 0 else float("inf")

    return {
        "img_adv": img_adv,
        "success": success,
        "success_iteration": success_iteration,
        "clean_faces": clean_faces,
        "adv_faces": count_faces(img_adv),
        "linf": linf_real,
        "l2": l2_real,
        "mse": mse_pert,
        "psnr": psnr
    }
