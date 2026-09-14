import torch

def pgd_attack(detector_wrapper, dsfd_net, mean_tensor, img_orig_tensor, epsilon, alpha, iterations, device):
    """
    Esegue l'attacco PGD mirato a eludere il rilevatore DSFD di DeepPrivacy2.
    """
    img_adv = img_orig_tensor.clone().detach()
    success = False
    success_iteration = None

    for i in range(iterations):
        img_adv.requires_grad_(True)
        
        # 1. Input normalizzato per DSFD
        input_net = img_adv.unsqueeze(0) - mean_tensor
        
        # 2. Forward pass
        net_out = dsfd_net(input_net, 0.0, 0.0)
        
        # 3. Calcolo Loss di evasione
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
        loss.backward()

        if img_adv.grad is None or torch.abs(img_adv.grad).sum().item() == 0:
            break

        grad_sign = img_adv.grad.sign()

        # 4. Aggiornamento PGD e Proiezione L-inf su scala 0-255
        with torch.no_grad():
            img_adv = img_adv - alpha * grad_sign
            eta = img_adv - img_orig_tensor
            eta = torch.clamp(eta, min=-epsilon, max=epsilon)
            img_adv = torch.clamp(img_orig_tensor + eta, min=0.0, max=255.0).detach()

        # 5. Check Evasione (ogni 5 iterazioni)
        if i % 5 == 0 or i == iterations - 1:
            detector_input = img_adv.detach().byte().float()
            with torch.no_grad():
                detections = detector_wrapper(detector_input)
            
            chk_faces = len(detections[0]) if (len(detections) > 0 and detections[0] is not None) else 0
            
            if chk_faces == 0:
                success = True
                success_iteration = i
                break

    return img_adv, success, success_iteration
