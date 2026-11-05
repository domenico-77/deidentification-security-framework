import torch
import numpy as np
import cv2
import dlib
import urllib.request
import bz2
from pathlib import Path
from attacks.base_attack import BaseAttack


class LandmarkSpoofingAttack(BaseAttack):
    """
    LandmarkSpoofingAttack con supporto region_mode.

    - region_mode="face" (default): perturba il rettangolo del volto (corretto per DP2)
    - region_mode="ring": perturba l'anello periferico attorno al volto (corretto per CIAGAN)

    L'attacco cerca una perturbazione che massimizzi lo spostamento dei landmark
    rilevati da dlib, portando (idealmente) all'evasione totale del detector
    (dlib non trova più il volto → pipeline restituisce l'input invariato).
    """

    def __init__(self, target=None, detector_wrapper=None, device: torch.device = None,
                 region_mode="face"):
        det = detector_wrapper if detector_wrapper is not None else (
            getattr(target, 'detector_wrapper', None) if target else None)
        super().__init__(detector=det, device=device)
        self.target = target
        self.region_mode = region_mode.lower()
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_shape_predictor()

    def _load_shape_predictor(self):
        predictor_filename = "shape_predictor_68_face_landmarks.dat"
        current_dir = Path(__file__).resolve().parent
        search_roots = [
            current_dir,
            current_dir.parent,
            Path.cwd(),
            Path("/kaggle/input"),
            Path("/kaggle/working"),
        ]

        for root in search_roots:
            if root.exists():
                found_files = list(root.glob(f"**/{predictor_filename}"))
                if found_files:
                    predictor_path = found_files[0]
                    try:
                        return dlib.shape_predictor(str(predictor_path))
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

    def _compute_score(self, candidate, orig_shape, y1, y2, x1, x2):
        if self.predictor is None or self.dlib_detector is None:
            return 0.0, None, None, False

        cand_np = candidate.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        cand_gray = cv2.cvtColor(cand_np, cv2.COLOR_RGB2GRAY)
        cand_rects = self.dlib_detector(cand_gray, 1)

        if not cand_rects:
            return 99999.0, cand_gray, None, True  # Evasione totale

        try:
            new_shape = self.predictor(cand_gray, cand_rects[0])
            shift_key = sum(
                abs(orig_shape.part(k).x - new_shape.part(k).x) +
                abs(orig_shape.part(k).y - new_shape.part(k).y)
                for k in range(17, 48)
            )
            total_shift = sum(
                abs(orig_shape.part(k).x - new_shape.part(k).x) +
                abs(orig_shape.part(k).y - new_shape.part(k).y)
                for k in range(68)
            )
            score = float(shift_key * 1.5 + total_shift)
            return score, cand_gray, new_shape, False
        except Exception:
            return 99999.0, cand_gray, None, True

    def _build_perturbation_mask(self, img_adv, y1, y2, x1, x2):
        """Costruisce la maschera in base al region_mode.

        - "face": rettangolo del volto (default, per DP2)
        - "ring": anello periferico attorno al volto, esclusa la faccia (per CIAGAN)
        """
        mask = torch.zeros_like(img_adv)

        if img_adv.dim() == 3:
            _, h, w = img_adv.shape
        else:
            _, _, h, w = img_adv.shape

        if self.region_mode == "ring":
            # Anello: box esterno allargato del 60% attorno al volto, meno il volto
            pad_h = int((y2 - y1) * 0.6)
            pad_w = int((x2 - x1) * 0.6)
            yy1 = max(0, y1 - pad_h)
            xx1 = max(0, x1 - pad_w)
            yy2 = min(h, y2 + pad_h)
            xx2 = min(w, x2 + pad_w)

            outer = torch.zeros_like(img_adv)
            outer[:, yy1:yy2, xx1:xx2] = 1.0

            face = torch.zeros_like(img_adv)
            face[:, y1:y2, x1:x2] = 1.0

            mask = (outer - face).clamp(0.0, 1.0)
        else:
            # "face": rettangolo del volto
            mask[:, y1:y2, x1:x2] = 1.0

        return mask

    def perturb(self, img_orig_tensor, epsilon=16.0, num_iter=50):
        img_adv = img_orig_tensor.clone().detach().to(self.device)
        img_orig_tensor = img_orig_tensor.to(self.device)

        success = False
        success_iteration = num_iter

        # dlib ha bisogno di HxWx3 uint8
        img_np = img_orig_tensor.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        rects = self.dlib_detector(gray, 1)

        if not rects or self.predictor is None:
            # Nessun volto rilevato già in partenza: "successo" banale
            return img_adv, True, 1

        det = rects[0]
        orig_shape = self.predictor(gray, det)

        x1, y1 = max(0, det.left()), max(0, det.top())
        x2, y2 = min(img_np.shape[1], det.right()), min(img_np.shape[0], det.bottom())

        best_candidate = img_adv.clone()
        current_perturbation = torch.zeros_like(img_adv)

        for i in range(num_iter):
            u = torch.sign(torch.randn_like(img_adv))

            # FIX: maschera target-aware (ring per CIAGAN, face per DP2)
            mask = self._build_perturbation_mask(img_adv, y1, y2, x1, x2)
            u = u * mask

            step_size = epsilon * 0.4

            pert_plus = torch.clamp(current_perturbation + step_size * u,
                                    min=-epsilon, max=epsilon)
            pert_minus = torch.clamp(current_perturbation - step_size * u,
                                     min=-epsilon, max=epsilon)

            cand_plus = torch.clamp(img_orig_tensor + pert_plus, min=0.0, max=255.0)
            cand_minus = torch.clamp(img_orig_tensor + pert_minus, min=0.0, max=255.0)

            score_plus, _, _, evaded_plus = self._compute_score(
                cand_plus, orig_shape, y1, y2, x1, x2)
            score_minus, _, _, evaded_minus = self._compute_score(
                cand_minus, orig_shape, y1, y2, x1, x2)

            if score_plus > score_minus:
                current_perturbation = pert_plus
                best_candidate = cand_plus
            else:
                current_perturbation = pert_minus
                best_candidate = cand_minus

            # Evasione totale (dlib non trova più alcun volto) → stop immediato
            if evaded_plus or evaded_minus:
                success = True
                success_iteration = i + 1
                break

        return best_candidate, success, success_iteration

    def attack(self, image_tensor: torch.Tensor, epsilon=16.0, **kwargs) -> torch.Tensor:
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
