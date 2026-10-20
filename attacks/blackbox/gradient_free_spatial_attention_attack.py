import bz2
import urllib.request
from pathlib import Path
import yaml

import cv2
import dlib
import numpy as np
import torch

from attacks.base_attack import BaseAttack


class GradientFreeSpatialAttentionAttack(BaseAttack):
    """
    Attacco black-box gradient-free con maschera spaziale e spinta direzionale
    basata sul gradiente dello score continuo della SVM del detector HOG.
    """

    def __init__(
        self,
        target=None,
        detector_wrapper=None,
        device: torch.device = None,
        num_iter: int = 500,
    ):
        det = detector_wrapper
        if det is None and target is not None:
            det = getattr(target, "detector_wrapper", None)

        super().__init__(detector=det, device=device)

        self.target = target
        self.default_num_iter = num_iter
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_shape_predictor()

    def _load_shape_predictor(self):
        filename = "shape_predictor_68_face_landmarks.dat"
        
        config_path = Path("configs/ciagan_paths.yaml")
        yaml_config = {}
        if config_path.exists():
            with open(config_path, "r") as f:
                yaml_config = yaml.safe_load(f).get("target", {})

        candidates = []
        if "repo_root" in yaml_config:
            candidates.append(Path(yaml_config["repo_root"]) / "source" / filename)

        candidates.extend([
            Path.cwd() / filename,
            Path(__file__).resolve().parent / filename,
            Path(__file__).resolve().parent.parent / filename,
            Path("/kaggle/input/ciagan-repository/ciagan/source") / filename,
        ])

        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            for base in [kaggle_input] + list(kaggle_input.glob("*/")):
                if base.is_dir():
                    candidates.append(base / "ciagan-repository" / "ciagan" / "source" / filename)
                    candidates.append(base / "ciagan" / "source" / filename)

        for path in candidates:
            if path and path.exists():
                try:
                    return dlib.shape_predictor(str(path))
                except (RuntimeError, OSError):
                    continue

        output_path = Path.cwd() / filename
        compressed_path = Path.cwd() / f"{filename}.bz2"
        url = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

        try:
            print(f"[INFO] Download automatico di {filename} in corso...")
            urllib.request.urlretrieve(url, str(compressed_path))

            with bz2.BZ2File(compressed_path, "rb") as source:
                with open(output_path, "wb") as destination:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        destination.write(chunk)

            compressed_path.unlink(missing_ok=True)
            return dlib.shape_predictor(str(output_path))

        except Exception as exc:
            print(
                "[WARN] Predictor dlib non disponibile: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    @staticmethod
    def _validate_image(image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("L'immagine deve essere un torch.Tensor.")

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                "Formato atteso [3, H, W]; ricevuto "
                f"{tuple(image.shape)}."
            )

        if image.shape[1] < 2 or image.shape[2] < 2:
            raise ValueError("Dimensioni dell'immagine non valide.")

        if not torch.isfinite(image).all():
            raise ValueError("L'immagine contiene NaN o valori infiniti.")

        low = image.min().item()
        high = image.max().item()

        if low < 0 or high > 255:
            raise ValueError(
                f"Range pixel non supportato: [{low:.4f}, {high:.4f}]."
            )

        normalized = high <= 1.0
        return normalized

    @staticmethod
    def _to_pixel_tensor(image, normalized):
        image = image.float()
        if normalized:
            image = image * 255.0
        return image.clamp(0.0, 255.0)

    @staticmethod
    def _to_original_scale(image_pixels, normalized):
        image = image_pixels.clamp(0.0, 255.0)
        if normalized:
            image = image / 255.0
        return image

    @staticmethod
    def _to_rgb_uint8(image_pixels):
        array = (
            image_pixels.detach()
            .float()
            .clamp(0.0, 255.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return np.rint(array).astype(np.uint8)

    @staticmethod
    def _to_gray(image_rgb):
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    def _create_attention_mask(self, img_shape, shape, det):
        height, width = img_shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)

        if shape is None:
            top = max(0, det.top())
            bottom = min(height, det.bottom() + 1)
            left = max(0, det.left())
            right = min(width, det.right() + 1)

            if bottom > top and right > left:
                mask[top:bottom, left:right] = 1.0
        else:
            indices = list(range(17, 48))
            points = np.array(
                [
                    (shape.part(i).x, shape.part(i).y)
                    for i in indices
                ],
                dtype=np.int32,
            )

            if len(points) >= 3:
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 1.0)

        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask = np.clip(mask, 0.0, 1.0)

        return torch.from_numpy(mask).unsqueeze(0).repeat(3, 1, 1)

    def _compute_score(self, candidate, orig_shape):
        """
        Usa .run() di dlib per estrarre rects e scores continui della SVM.
        Restituisce (score, detector_evasion).
        """
        if self.dlib_detector is None:
            return float("inf"), False

        try:
            candidate_rgb = self._to_rgb_uint8(candidate)
            candidate_gray = self._to_gray(candidate_rgb)
            dets, scores, _ = self.dlib_detector.run(candidate_gray, 1, 0.0)
        except Exception:
            return float("inf"), False

        if len(dets) == 0:
            return float("inf"), True  # Evasione totale

        try:
            min_score = float(min(scores)) if len(scores) > 0 else 0.0
            score = float(min_score)

            if not np.isfinite(score):
                return float("inf"), False

            return score, False
        except Exception:
            return float("inf"), False

    def perturb(self, img_orig_tensor, epsilon=16.0, num_iter=None):
        if num_iter is None:
            num_iter = self.default_num_iter

        if not isinstance(num_iter, int) or num_iter < 1:
            raise ValueError("num_iter deve essere un intero positivo.")

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon deve essere un valore finito >= 0.")

        input_normalized = self._validate_image(img_orig_tensor)

        original_scale = img_orig_tensor.detach().to(self.device).float()
        original_pixels = self._to_pixel_tensor(
            original_scale, input_normalized
        )

        if self.predictor is None or self.dlib_detector is None:
            return original_scale.clone(), False, 0

        original_rgb = self._to_rgb_uint8(original_pixels)
        original_gray = self._to_gray(original_rgb)

        try:
            rects = self.dlib_detector(original_gray, 1)
        except Exception:
            return original_scale.clone(), False, 0

        if len(rects) == 0:
            return original_scale.clone(), False, 0

        try:
            original_shape = self.predictor(original_gray, rects[0])
        except Exception:
            return original_scale.clone(), False, 0

        attention_mask = self._create_attention_mask(
            original_rgb.shape,
            original_shape,
            rects[0],
        ).to(device=self.device, dtype=torch.float32)

        if not torch.isfinite(attention_mask).all():
            return original_scale.clone(), False, 0

        current_perturbation = torch.zeros_like(original_pixels)
        best_candidate = original_pixels.clone()
        best_score = float("inf")
        
        successful_candidates = []
        step_size = float(epsilon) * 0.15

        # Vettore di momento / spinta direzionale basato sullo score
        directional_momentum = torch.zeros_like(original_pixels)

        for iteration in range(num_iter):
            # Combina la componente stocastica con la spinta direzionale accumulata
            random_dir = torch.sign(torch.randn_like(original_pixels)) * attention_mask
            direction = torch.sign(random_dir + 0.5 * directional_momentum) * attention_mask

            perturbation_plus = torch.clamp(
                current_perturbation + step_size * direction,
                min=-epsilon,
                max=epsilon,
            )

            perturbation_minus = torch.clamp(
                current_perturbation - step_size * direction,
                min=-epsilon,
                max=epsilon,
            )

            candidate_plus = torch.clamp(
                original_pixels + perturbation_plus,
                min=0.0,
                max=255.0,
            )

            candidate_minus = torch.clamp(
                original_pixels + perturbation_minus,
                min=0.0,
                max=255.0,
            )

            score_plus, evaded_plus = self._compute_score(
                candidate_plus, original_shape
            )
            score_minus, evaded_minus = self._compute_score(
                candidate_minus, original_shape
            )

            if evaded_plus:
                l2_dist = torch.norm(candidate_plus - original_pixels).item()
                successful_candidates.append((l2_dist, candidate_plus, iteration + 1))

            if evaded_minus:
                l2_dist = torch.norm(candidate_minus - original_pixels).item()
                successful_candidates.append((l2_dist, candidate_minus, iteration + 1))

            # Aggiornamento della spinta direzionale in base a quale verso riduce lo score della SVM
            if score_plus < score_minus:
                chosen_perturbation = perturbation_plus
                chosen_candidate = candidate_plus
                chosen_score = score_plus
                # Se il verso plus ha migliorato lo score, rinforziamo la direzione
                directional_momentum = 0.7 * directional_momentum + 0.3 * direction
            else:
                chosen_perturbation = perturbation_minus
                chosen_candidate = candidate_minus
                chosen_score = score_minus
                # Se il verso minus è stato migliore, spingiamo nella direzione opposta
                directional_momentum = 0.7 * directional_momentum - 0.3 * direction

            current_perturbation = chosen_perturbation

            if chosen_score < best_score:
                best_score = chosen_score
                best_candidate = chosen_candidate.clone()

        if successful_candidates:
            successful_candidates.sort(key=lambda x: x[0])
            _, best_success_cand, success_iter = successful_candidates[0]
            result = self._to_original_scale(best_success_cand, input_normalized)
            return result.detach(), True, success_iter

        result = self._to_original_scale(
            best_candidate, input_normalized
        )

        return result.detach(), False, num_iter

    def attack(
        self,
        image_tensor: torch.Tensor,
        epsilon=16.0,
        **kwargs,
    ) -> torch.Tensor:
        image_adv, _, _ = self.perturb(
            image_tensor,
            epsilon=epsilon,
            **kwargs,
        )
        return image_adv
