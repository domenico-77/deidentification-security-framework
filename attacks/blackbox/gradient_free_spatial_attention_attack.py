import torch
import numpy as np
import cv2
import dlib
import urllib.request
import bz2
from pathlib import Path
from attacks.base_attack import BaseAttack

class GradientFreeSpatialAttentionAttack(BaseAttack):
    """
    Gradient-Free Spatial Attention Attack (Black-Box Mode).
    Mira ad azzerare le detection di dlib concentrando la perturbazione sulle regioni di transizione critiche
    per il descrittore HOG (occhi, sopracciglia, naso), preservando al contempo la struttura globale
    per garantire un'alta similarità biometrica con ArcFace.
    """
    def __init__(self, target=None, detector_wrapper=None, device: torch.device = None):
        det = detector_wrapper if detector_wrapper is not None else (getattr(target, 'detector_wrapper', None) if target else None)
        super().__init__(detector=det, device=device)
        self.target = target
        
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_shape_predictor()

    def _load_shape_predictor(self):
        predictor_filename = "shape_predictor_68_face_landmarks.dat"
        current_dir = Path(__file__).resolve().parent
        search_roots = [
            current_dir,
            current_dir.parent,
            current_dir.parent.parent,
            Path.cwd(),
            Path("/kaggle/input"),
            Path("/kaggle/working"),
        ]

        for root in search_roots:
            if root.exists():
                found_files = list(root.glob(f"**/{predictor_filename}"))
                if found_files:
                    try:
                        return dlib.shape_predictor(str(found_files[0]))
                    except Exception:
                        pass

        fallback_path = Path(predictor_filename)
        bz2_path = fallback_path.with_suffix(".dat.bz2")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        try:
            urllib.request.urlretrieve(url, bz2_path)
            with bz2.BZ2File(bz2_path) as fr, open(fallback_path, "wb") as fw:
                fw.write(fr.read())
            bz2_path.unlink(missing_ok=True)
            return dlib.shape_predictor(str(fallback_path))
        except Exception:
            return None

    def _create_attention_mask(self, img_shape, shape, det):
        mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
        
        if shape is None:
            mask[max(0, det.top()):min(img_shape[0], det.bottom()), 
                 max(0, det.left()):min(img_shape[1], det.right())] = 1.0
            return torch.from_numpy(mask).unsqueeze(0).repeat(3, 1, 1)

        key_indices = list(range(17, 48))
        points = np.array([(shape.part(i).x, shape.part(i).y) for i in key_indices], dtype=np.int32)
        
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 1.0)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return torch.from_numpy(mask).unsqueeze(0).repeat(3, 1, 1)

    def perturb(self, img_orig_tensor, epsilon=12.0, num_iter=50):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)
        
        success = False
        success_iteration = num_iter

        img_np = img_orig_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        rects = self.dlib_detector(gray, 1)

        if not rects or self.predictor is None:
            return img_adv, True, 1

        det = rects[0]
        orig_shape = self.predictor(gray, det)
        
        attention_mask = self._create_attention_mask(img_np.shape, orig_shape, det).to(self.device)
        
        best_candidate = img_adv.clone()
        current_perturbation = torch.zeros_like(img_adv)

        for i in range(num_iter):
            u = torch.sign(torch.randn_like(img_adv)) * attention_mask
            step_size = epsilon * 0.35
            
            pert_plus = torch.clamp(current_perturbation + step_size * u, min=-epsilon, max=epsilon)
            pert_minus = torch.clamp(current_perturbation - step_size * u, min=-epsilon, max=epsilon)
            
            cand_plus = torch.clamp(img_orig_tensor + pert_plus, min=0.0, max=255.0)
            cand_minus = torch.clamp(img_orig_tensor + pert_minus, min=0.0, max=255.0)

            cand_plus_np = cand_plus.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            rects_plus = self.dlib_detector(cv2.cvtColor(cand_plus_np, cv2.COLOR_RGB2GRAY), 1)
            evaded_plus = len(rects_plus) == 0

            cand_minus_np = cand_minus.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            rects_minus = self.dlib_detector(cv2.cvtColor(cand_minus_np, cv2.COLOR_RGB2GRAY), 1)
            evaded_minus = len(rects_minus) == 0

            if evaded_plus:
                current_perturbation = pert_plus
                best_candidate = cand_plus
                success = True
                success_iteration = i + 1
                break
            elif evaded_minus:
                current_perturbation = pert_minus
                best_candidate = cand_minus
                success = True
                success_iteration = i + 1
                break
            else:
                current_perturbation = pert_plus
                best_candidate = cand_plus

        return best_candidate, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=12.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
