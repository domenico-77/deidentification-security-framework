import bz2
import urllib.request
from pathlib import Path
yaml_mod = None
try:
    import yaml
    yaml_mod = yaml
except ImportError:
    pass

import cv2
import dlib
import numpy as np
import torch
import torch.nn.functional as F

from attacks.base_attack import BaseAttack


class GradientFreeSpatialAttentionAttack(BaseAttack):
    """
    Attacco black-box con maschera spaziale periferica.

    v4: ricerca a bassa frequenza. La perturbazione e' parametrizzata su una
    griglia ridotta (grid_size x grid_size) e poi upsamplingata: lo spazio di
    ricerca scende da ~10^5-10^6 dimensioni a poche centinaia, cosi' la
    ricerca random+momentum converge entro il budget di query. Le
    perturbazioni a bassa frequenza aggrediscono anche i gradienti interni
    dell'HOG che l'anello di contorno non copre. La similarita' viene
    ripristinata dalla fase di shrink operante nello stesso spazio.
    """

    GRID_SIZE = 10          # dimensioni della griglia di ricerca
    BAND_RATIO = 0.20       # banda dell'anello di contorno
    SIM_BUDGET_QUERIES = 1000

    def __init__(
        self,
        target=None,
        detector_wrapper=None,
        device: torch.device = None,
        num_iter: int = 500,
        num_restarts: int = 10,
    ):
        det = detector_wrapper
        if det is None and target is not None:
            det = getattr(target, "detector_wrapper", None)

        super().__init__(detector=det, device=device)

        self.target = target
        self.default_num_iter = num_iter
        self.num_restarts = num_restarts
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_shape_predictor()

    def _load_shape_predictor(self):
        filename = "shape_predictor_68_face_landmarks.dat"

        config_path = Path("configs/ciagan_paths.yaml")
        yaml_config = {}
        if config_path.exists() and yaml_mod is not None:
            with open(config_path, "r") as f:
                yaml_config = yaml_mod.safe_load(f).get("target", {})

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
            outer_indices = list(range(0, 27))
            outer_points = np.array(
                [(shape.part(i).x, shape.part(i).y) for i in outer_indices],
                dtype=np.int32,
            )
            if len(outer_points) >= 3:
                hull = cv2.convexHull(outer_points)
                cv2.fillConvexPoly(mask, hull, 1.0)

                face_w = max(1, hull[:, 0, 0].max() - hull[:, 0, 0].min())
                band = max(3, int(face_w * self.BAND_RATIO))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (band, band)
                )
                inner = cv2.erode(mask, kernel, iterations=1)
                mask = np.clip(mask - inner, 0.0, 1.0)

                inner_mask = np.zeros((height, width), dtype=np.float32)
                inner_indices = list(range(27, 68))
                inner_points = np.array(
                    [(shape.part(i).x, shape.part(i).y) for i in inner_indices],
                    dtype=np.int32,
                )
                if len(inner_points) >= 3:
                    inner_hull = cv2.convexHull(inner_points)
                    cv2.fillConvexPoly(inner_mask, inner_hull, 1.0)
                    mask = np.clip(mask - inner_mask, 0.0, 1.0)

        mask = cv2.GaussianBlur(mask, (7, 7), 1.5)
        mask = np.clip(mask, 0.0, 1.0)

        return torch.from_numpy(mask).unsqueeze(0).repeat(3, 1, 1)

    def _compute_score(self, candidate, orig_shape):
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

    def _grid_to_delta(self, theta, attention_mask):
        """
        Converte i coefficienti della griglia (grid x grid) in una
        perturbazione a piena risoluzione, limitata dalla maschera.
        """
        g = theta.unsqueeze(0).unsqueeze(0)  # [1, 1, g, g]
        up = F.interpolate(
            g,
            size=(attention_mask.shape[1], attention_mask.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # [1, H, W]
        delta = up * attention_mask[:1]
        return delta.repeat(3, 1, 1)

    def _shrink_theta(self, original_pixels, theta, attention_mask, max_queries):
        """
        v4: riduce la perturbazione evasante (backoff moltiplicativo + dropout
        di celle della griglia) mantenendo l'evasione, per alzare la
        similarita' con l'originale.
        """
        queries = 0
        best_theta = theta.clone()
        current = theta.clone()
        grid = theta.shape[0]

        for factor in (0.75, 0.5, 0.35, 0.2, 0.1):
            if queries >= max_queries:
                break
            trial_theta = current * factor
            trial = torch.clamp(
                original_pixels + self._grid_to_delta(trial_theta, attention_mask),
                0.0, 255.0,
            )
            _, evaded = self._compute_score(trial, None)
            queries += 1
            if evaded:
                best_theta = trial_theta
                current = trial_theta

        for _ in range(12):
            if queries >= max_queries:
                break
            keep = (torch.rand(grid, grid, device=theta.device) > 0.3).float()
            trial_theta = current * keep
            trial = torch.clamp(
                original_pixels + self._grid_to_delta(trial_theta, attention_mask),
                0.0, 255.0,
            )
            _, evaded = self._compute_score(trial, None)
            queries += 1
            if evaded:
                best_theta = trial_theta
                current = trial_theta

        return best_theta, queries

    def perturb(self, img_orig_tensor, epsilon=64.0, num_iter=None):
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

        grid = self.GRID_SIZE
        total_queries = 0
        successful_theta = None
        success_queries = 0
        global_best_score = float("inf")
        global_best_theta = torch.zeros(grid, grid, device=self.device)

        iters_per_restart = max(10, num_iter // self.num_restarts)

        for restart in range(self.num_restarts):
            # v4: se abbiamo gia' un'evasione, il budget residuo va allo
            # shrink (che migliora la similarita'), non ad altre ricerche.
            if successful_theta is not None:
                break

            theta = torch.zeros(grid, grid, device=self.device)
            momentum = torch.zeros(grid, grid, device=self.device)
            restart_best_score = float("inf")
            restart_best_theta = theta.clone()

            step_size = 1.0  # step nello spazio normalizzato della griglia

            for iteration in range(iters_per_restart):
                total_queries += 2

                random_dir = torch.randn(grid, grid, device=self.device)
                direction = torch.sign(
                    random_dir + 0.6 * momentum
                )

                theta_plus = torch.clamp(
                    theta + step_size * direction, min=-1.0, max=1.0
                )
                theta_minus = torch.clamp(
                    theta - step_size * direction, min=-1.0, max=1.0
                )

                candidate_plus = torch.clamp(
                    original_pixels
                    + epsilon * self._grid_to_delta(theta_plus, attention_mask),
                    0.0, 255.0,
                )
                candidate_minus = torch.clamp(
                    original_pixels
                    + epsilon * self._grid_to_delta(theta_minus, attention_mask),
                    0.0, 255.0,
                )

                score_plus, evaded_plus = self._compute_score(
                    candidate_plus, original_shape
                )
                score_minus, evaded_minus = self._compute_score(
                    candidate_minus, original_shape
                )

                if evaded_plus:
                    successful_theta = theta_plus.clone()
                    success_queries = total_queries
                    break
                if evaded_minus:
                    successful_theta = theta_minus.clone()
                    success_queries = total_queries
                    break

                if score_plus <= score_minus:
                    theta = theta_plus
                    chosen_score = score_plus
                    momentum = 0.6 * momentum + 0.4 * direction
                else:
                    theta = theta_minus
                    chosen_score = score_minus
                    momentum = 0.6 * momentum - 0.4 * direction

                if chosen_score < restart_best_score:
                    restart_best_score = chosen_score
                    restart_best_theta = theta.clone()

            if restart_best_score < global_best_score:
                global_best_score = restart_best_score
                global_best_theta = restart_best_theta.clone()

        if successful_theta is not None:
            # v4: fase di shrink entro il budget residuo (tetto 1000 query).
            budget = max(0, self.SIM_BUDGET_QUERIES - success_queries)
            shrunk_theta, shrink_queries = self._shrink_theta(
                original_pixels, successful_theta, attention_mask, budget
            )
            success_queries += shrink_queries

            result = torch.clamp(
                original_pixels
                + epsilon * self._grid_to_delta(shrunk_theta, attention_mask),
                0.0, 255.0,
            )
            result = self._to_original_scale(result, input_normalized)
            return result.detach(), True, success_queries

        # Fallback: miglior score trovato (nessuna evasione)
        result = torch.clamp(
            original_pixels
            + epsilon * self._grid_to_delta(global_best_theta, attention_mask),
            0.0, 255.0,
        )
        result = self._to_original_scale(result, input_normalized)

        return result.detach(), False, total_queries

    def attack(
        self,
        image_tensor: torch.Tensor,
        epsilon=64.0,
        **kwargs,
    ) -> torch.Tensor:
        image_adv, _, _ = self.perturb(
            image_tensor,
            epsilon=epsilon,
            **kwargs,
        )
        return image_adv
