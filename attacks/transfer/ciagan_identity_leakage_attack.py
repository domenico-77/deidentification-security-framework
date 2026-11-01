import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import dlib

from attacks.base_attack import BaseAttack

GEN_HW = (128, 128)
BAND_RATIO = 0.20
DEFAULT_NUM_ITER = 100
DEFAULT_ALPHA = 8.0
LNDM_MAX_SHIFT = 6.0
LNDM_SIGMA = 2.0
SUCCESS_SIM_THRESHOLD = 0.65
SUCCESS_IMPROVEMENT_MARGIN = 0.10

_GENERATOR_PATHS = [
    "/kaggle/input/datasets/domenicovicenti/ciagan-models/modelG_ciagan.pth",
    "/kaggle/working/modelG_ciagan.pth",
    "modelG_ciagan.pth",
]


def _ensure_facenet_installed():
    try:
        import facenet_pytorch  # noqa: F401
        return True
    except ImportError:
        pass
    print("[CiaganLeakage] facenet-pytorch non trovato. Installazione con --no-deps...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "facenet-pytorch",
             "--no-deps", "--quiet", "--disable-pip-version-check"]
        )
    except Exception as e:
        print(f"[CiaganLeakage] pip install fallito: {type(e).__name__}: {e}")
        return False
    try:
        import facenet_pytorch  # noqa: F401
        print("[CiaganLeakage] facenet-pytorch installato.")
        return True
    except ImportError as e:
        print(f"[CiaganLeakage] import dopo install fallito: {e}")
        return False


def _find_first(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _to_hwc_numpy(img_tensor_or_arr):
    if isinstance(img_tensor_or_arr, torch.Tensor):
        arr = img_tensor_or_arr.detach().cpu().numpy()
    else:
        arr = np.array(img_tensor_or_arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.clip(arr, 0, 255).astype(np.uint8)


class CiaganIdentityLeakageAttack(BaseAttack):
    """Identity Leakage Attack contro CIAGAN.

    FIX: Accetta `mean_tensor=None` e `**kwargs` per compatibilità con factory generiche
    (es. eval_reidentification_transfer.py) che passano parametri extra.

    FIX 2: Ripristinato il calcolo di `baseline_sim = cos(e_CIAGAN(clean), e_CIAGAN(adv))`
    e la sua propagazione via `_last_baseline_sim`, che era stata rimossa in un
    refactoring precedente (causando `baseline_vs_adv = 0.0` costante).
    """

    def __init__(self, target=None, detector_wrapper=None, device: torch.device = None,
                 generator_path=None, identity_index=0, arcface_model=None,
                 use_real_target_oracle=True, mean_tensor=None, **kwargs):
        self.mean_tensor = mean_tensor  # compat, ignorato
        if kwargs:
            print(f"[CiaganLeakage] __init__: parametri extra ignorati: {list(kwargs.keys())}")

        det = detector_wrapper if detector_wrapper is not None else (
            getattr(target, "detector_wrapper", None) if target else None)
        super().__init__(detector=det, device=device)

        self.target = target
        self.identity_index = identity_index
        self._surrogate_type = None
        self._surrogate_input_size = 160
        self._surrogate_norm_mean = (0.5, 0.5, 0.5)
        self._surrogate_norm_std = (0.5, 0.5, 0.5)

        self._id_surrogate = arcface_model
        if self._id_surrogate is None and target is not None:
            self._id_surrogate = self._find_identity_oracle(target)
        if self._id_surrogate is None:
            self._id_surrogate = self._load_default_arcface()

        if self._id_surrogate is None:
            print("[CiaganLeakage] FATAL: nessun oracolo biometrico disponibile.")
        else:
            print(f"[CiaganLeakage] oracolo = {type(self._id_surrogate).__name__} "
                  f"({self._surrogate_type})")

        self.dlib_detector = dlib.get_frontal_face_detector()
        self.predictor = self._load_shape_predictor()
        self._load_generator(generator_path)

    @staticmethod
    def _find_identity_oracle(target):
        for attr in ("arcface", "id_model", "recognizer", "face_recognizer",
                     "identity_model", "embedder", "id_net", "recog", "oracle",
                     "surrogate", "face_net", "recognition_model", "feature_extractor"):
            cand = getattr(target, attr, None)
            if cand is not None and not isinstance(cand, str):
                return cand
        try:
            for k, v in vars(target).items():
                if isinstance(v, torch.nn.Module):
                    if any(t in k.lower() for t in ("arc", "recog", "ident", "face", "embed")):
                        return v
        except Exception:
            pass
        return None

    def _load_default_arcface(self):
        if _ensure_facenet_installed():
            try:
                from facenet_pytorch import InceptionResnetV1
                m = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                for p in m.parameters():
                    p.requires_grad_(False)
                self._surrogate_type = "torch"
                self._surrogate_input_size = 160
                print("[CiaganLeakage] loaded facenet_pytorch InceptionResnetV1 (differentiable)")
                return m
            except Exception as e:
                print(f"[CiaganLeakage] facenet load failed: {type(e).__name__}: {e}")

        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='antelopev2',
                               providers=['CUDAExecutionProvider' if self.device.type == 'cuda'
                                          else 'CPUExecutionProvider'])
            app.prepare(ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))
            rec = app.models.get('recognition', None)
            if rec is not None:
                self._surrogate_type = "onnx"
                self._surrogate_input_size = 112
                print("[CiaganLeakage] loaded insightface ArcFaceONNX (NON-differentiable)")
                return rec
        except Exception:
            pass

        try:
            import torchvision.models as tvm
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
            m.fc = torch.nn.Identity()
            m.eval().to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
            self._surrogate_type = "torch"
            self._surrogate_input_size = 224
            self._surrogate_norm_mean = (0.485, 0.456, 0.406)
            self._surrogate_norm_std = (0.229, 0.224, 0.225)
            print("[CiaganLeakage] loaded torchvision ResNet50 (proxy generico)")
            return m
        except Exception:
            return None

    def _load_shape_predictor(self):
        filename = "shape_predictor_68_face_landmarks.dat"
        current_dir = Path(__file__).resolve().parent
        for root in [current_dir, current_dir.parent, current_dir.parent.parent,
                     Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working")]:
            if root.exists():
                found = list(root.glob(f"**/{filename}"))
                if found:
                    try:
                        return dlib.shape_predictor(str(found[0]))
                    except Exception:
                        pass
        return None

    def _load_generator(self, generator_path=None):
        path = generator_path or _find_first(_GENERATOR_PATHS)
        if path is None:
            self.netG = None
            return
        for cand in [
            "/kaggle/input/datasets/domenicovicenti/ciagan-repository/ciagan/source",
            "/kaggle/input/datasets/domenicovicenti/ciagan-repository/ciagan/source/arch",
            "/kaggle/working/source", "/kaggle/working/source/arch",
            str(Path(__file__).resolve().parent.parent / "source"),
            str(Path(__file__).resolve().parent.parent / "source" / "arch"),
        ]:
            if os.path.isdir(cand) and cand not in sys.path:
                sys.path.insert(0, cand)
        try:
            from arch_unet_flex import Generator
            netG = Generator().to(self.device)
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            netG.load_state_dict(state)
            netG.eval()
            for p in netG.parameters():
                p.requires_grad_(False)
            self.netG = netG
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"[CiaganLeakage] generator load failed: {msg}")
            if "dlopen" in msg or "undefined symbol" in msg:
                print("       Riavvia il runtime Kaggle.")
            self.netG = None

    def _build_mask_and_landmarks(self, img_uint8_hwc):
        hwc = _to_hwc_numpy(img_uint8_hwc)
        h, w = hwc.shape[:2]
        gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY) if hwc.ndim == 3 else hwc
        rects = self.dlib_detector(gray, 1)
        mask = np.zeros((h, w), dtype=np.float32)
        lndm = np.zeros((h, w), dtype=np.float32)
        pts = None
        if rects and self.predictor is not None:
            shape = self.predictor(gray, rects[0])
            pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)],
                           dtype=np.float32)
            hull = cv2.convexHull(pts.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)
            for (x, y) in pts.astype(np.int32):
                if 0 <= y < h and 0 <= x < w:
                    lndm[y, x] = 1.0
        else:
            cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.30), int(h * 0.40)),
                        0, 0, 360, 1.0, -1)
            angles = np.linspace(0, 2 * np.pi, 68, endpoint=False).astype(np.float32)
            pts = np.stack([
                w * 0.5 + w * 0.30 * np.cos(angles),
                h * 0.5 + h * 0.40 * np.sin(angles),
            ], axis=1).astype(np.float32)
            for (x, y) in pts.astype(np.int32):
                if 0 <= y < h and 0 <= x < w:
                    lndm[y, x] = 1.0
        return mask, lndm, pts

    def _ring_mask(self, x01_chw, im_msk_1hw):
        m = im_msk_1hw[0, 0].detach().cpu().numpy()
        m_u8 = (m * 255).astype(np.uint8)
        h, w = m_u8.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        hull = cv2.dilate(m_u8, kernel, iterations=1)
        erosion_k = max(5, int(2 * BAND_RATIO * min(h, w)))
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_k, erosion_k))
        inner = cv2.erode(hull, kernel_e, iterations=1)
        ring = cv2.subtract(hull, inner).astype(np.float32) / 255.0
        return torch.from_numpy(np.clip(ring, 0.0, 1.0)).to(self.device).unsqueeze(0).unsqueeze(0)

    def _identity_for_attack(self, img_tensor):
        for attr in ("current_identity", "identity_index", "get_identity_index"):
            obj = getattr(self.target, attr, None) if self.target is not None else None
            if obj is not None:
                try:
                    val = obj() if callable(obj) else obj
                    if isinstance(val, (int, np.integer)):
                        return (int(val) + 1) % 2
                except Exception:
                    pass
        return (self.identity_index + 1) % 2

    def _get_onehot(self, identity_index):
        oh = torch.zeros(1, 1200, device=self.device)
        val = 1.0 if identity_index == 0 else -1.0
        oh[0, :600] = val
        oh[0, 600:] = -val
        return oh

    def _compute_face_crop(self, x01_chw_1):
        hwc = _to_hwc_numpy(x01_chw_1[0] * 255.0)
        gray = cv2.cvtColor(hwc, cv2.COLOR_RGB2GRAY) if hwc.ndim == 3 else hwc
        rects = self.dlib_detector(gray, 1)
        if rects:
            rect = rects[0]
            pad = int(0.30 * max(rect.width(), rect.height()))
            x1 = max(0, rect.left() - pad)
            y1 = max(0, rect.top() - pad)
            x2 = min(hwc.shape[1], rect.right() + pad)
            y2 = min(hwc.shape[0], rect.bottom() + pad)
            return (x1, y1, x2, y2)
        h, w = hwc.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        half = side // 2
        return (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))

    def _embed_surrogate(self, x01_chw_1, crop_rect=None):
        if self._id_surrogate is None:
            return None
        try:
            if crop_rect is None:
                crop_rect = self._compute_face_crop(x01_chw_1)
            x1, y1, x2, y2 = crop_rect
            if x2 <= x1 or y2 <= y1:
                return None
            model = self._id_surrogate
            if self._surrogate_type == "onnx" or hasattr(model, "get_feat"):
                crop_t = x01_chw_1[:, :, y1:y2, x1:x2]
                hwc = _to_hwc_numpy(crop_t[0] * 255.0)
                bgr = cv2.cvtColor(cv2.resize(hwc, (112, 112)), cv2.COLOR_RGB2BGR)
                emb = model.get_feat(bgr)
                emb = torch.from_numpy(np.asarray(emb)).to(self.device)
                return F.normalize(emb.float(), dim=-1)
            crop_t = x01_chw_1[:, :, y1:y2, x1:x2]
            size = self._surrogate_input_size
            crop_t = F.interpolate(crop_t, size=(size, size),
                                   mode="bilinear", align_corners=False)
            mean = torch.tensor(self._surrogate_norm_mean, device=self.device).view(1, 3, 1, 1)
            std = torch.tensor(self._surrogate_norm_std, device=self.device).view(1, 3, 1, 1)
            crop_t = (crop_t - mean) / std
            emb = model(crop_t)
            if isinstance(emb, tuple):
                emb = emb[0]
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, device=self.device)
            if emb.numel() == 0:
                return None
            return F.normalize(emb.float(), dim=-1)
        except Exception as e:
            print(f"[CiaganLeakage._embed_surrogate] {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _render_landmarks_soft(pts_xy, offsets, H, W, sigma=LNDM_SIGMA):
        device = offsets.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        px = (pts_xy[:, 0] + offsets[:, 0]).view(-1, 1, 1)
        py = (pts_xy[:, 1] + offsets[:, 1]).view(-1, 1, 1)
        dx = xx.unsqueeze(0) - px
        dy = yy.unsqueeze(0) - py
        blobs = torch.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))
        canvas = blobs.max(dim=0)[0]
        return canvas.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    def perturb(self, img_orig_tensor, epsilon=None, num_iter=DEFAULT_NUM_ITER,
                alpha=DEFAULT_ALPHA, **kwargs):
        if epsilon is None:
            epsilon = getattr(self, "epsilon", 32.0)
        eps01 = float(epsilon) / 255.0
        a01 = float(alpha) / 255.0
        print(f"[CiaganLeakage] perturb(eps={epsilon} iters={num_iter})")

        self._last_oracle_sim = 0.0
        self._last_baseline_sim = 0.0       # <-- baseline_vs_adv (cos tra i due output CIAGAN)
        self._last_baseline_orig_sim = 0.0  # <-- baseline (cos tra e_orig e output CIAGAN clean)

        if self._id_surrogate is None or self.netG is None:
            print("[CiaganLeakage] PREREQUISITI MANCANTI: ritorno originale")
            return img_orig_tensor.clone().detach(), False, 0

        img = img_orig_tensor.clone().detach().to(self.device)
        if img.dim() == 4:
            img = img.squeeze(0)
        if img.max() <= 1.5:
            img = img * 255.0
        x0_full = img.detach().float()

        identity_idx = self._identity_for_attack(x0_full)
        x_res = F.interpolate(x0_full.unsqueeze(0) / 255.0, size=GEN_HW,
                              mode="bilinear", align_corners=False)
        arr = (x_res[0].detach().permute(1, 2, 0).cpu().numpy() * 255.0)
        mask_np, _lndm_np, pts_np = self._build_mask_and_landmarks(arr)
        im_msk = torch.from_numpy(mask_np).to(self.device).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        ring = self._ring_mask(x_res, im_msk)
        H, W = GEN_HW
        pts_xy = torch.tensor(pts_np, dtype=torch.float32, device=self.device)
        face_crop_rect = self._compute_face_crop(x_res)

        e0_ref = self._embed_surrogate(x_res, crop_rect=face_crop_rect)
        if e0_ref is None:
            print("[CiaganLeakage] FATAL: e0_ref=None")
            return x0_full.clone(), False, 0
        e0_ref = e0_ref.detach()

        labels_one_hot = self._get_onehot(identity_idx)
        delta = torch.zeros_like(x_res, requires_grad=True)
        lndm_off = torch.zeros_like(pts_xy, requires_grad=True)
        step_ring = a01
        step_lndm = LNDM_MAX_SHIFT * 1.5 / max(num_iter, 1)

        for it in range(num_iter):
            x_adv_t = torch.clamp(x_res + delta, 0.0, 1.0)
            im_lndm = self._render_landmarks_soft(pts_xy, lndm_off, H, W)
            inp = torch.cat((im_lndm, x_adv_t * (1.0 - im_msk)), dim=1)
            im_gen = self.netG(inp, onehot=labels_one_hot)
            im_gen_t = im_gen[0] if isinstance(im_gen, (list, tuple)) else im_gen
            out_anon = im_gen_t * im_msk + x_adv_t * (1.0 - im_msk)
            ea_pipe = self._embed_surrogate(out_anon, crop_rect=face_crop_rect)
            if ea_pipe is None:
                leakage_loss = torch.zeros(1, device=self.device, requires_grad=True)
            else:
                cos_sim = F.cosine_similarity(e0_ref, ea_pipe, dim=-1).mean()
                leakage_loss = 1.0 - cos_sim
            fid_ring = ((x_adv_t - x_res) ** 2 * ring).sum() / (ring.sum() + 1e-8)
            fid_lndm = (lndm_off ** 2).mean()
            loss = 20.0 * leakage_loss + 1.0 * fid_ring + 0.01 * fid_lndm
            loss.backward()
            if it == 0:
                g_ring = delta.grad.abs().sum().item() if delta.grad is not None else 0.0
                g_lndm = lndm_off.grad.abs().sum().item() if lndm_off.grad is not None else 0.0
                print(f"[CiaganLeakage] iter0 |g_ring|={g_ring:.3e} |g_lndm|={g_lndm:.3e}")
            with torch.no_grad():
                if delta.grad is not None:
                    g = delta.grad * ring
                    delta.data = torch.clamp(delta.data - step_ring * torch.sign(g),
                                             -eps01, eps01)
                    delta.grad.zero_()
                if lndm_off.grad is not None:
                    lndm_off.data = torch.clamp(
                        lndm_off.data - step_lndm * torch.sign(lndm_off.grad),
                        -LNDM_MAX_SHIFT, LNDM_MAX_SHIFT,
                    )
                    lndm_off.grad.zero_()

        delta_up = F.interpolate(delta.detach(), size=x0_full.shape[-2:],
                                  mode="bilinear", align_corners=False).squeeze(0)
        out_img = torch.clamp(x0_full + delta_up * 255.0, 0.0, 255.0)

        linf = (out_img - x0_full).abs().max().item()
        print(f"[CiaganLeakage] linf={linf:.2f}/{epsilon}")

        # =====================================================================
        # VALUTAZIONE FINALE: 3 metriche biometriche
        # =====================================================================
        final_sim = 0.0           # cos(e_orig,       e_CIAGAN(adv))
        baseline_orig_sim = 0.0   # cos(e_orig,       e_CIAGAN(clean))
        baseline_sim = 0.0        # cos(e_CIAGAN(clean), e_CIAGAN(adv))
        ea_final = None
        e_base = None
        with torch.no_grad():
            # ---- (a) output adversarial ----
            x_final = torch.clamp(x_res + delta.detach(), 0.0, 1.0)
            im_lndm_f = self._render_landmarks_soft(pts_xy, lndm_off.detach(), H, W)
            inp_adv = torch.cat((im_lndm_f, x_final * (1.0 - im_msk)), dim=1)
            im_gen_adv = self.netG(inp_adv, onehot=labels_one_hot)
            im_gen_adv_t = im_gen_adv[0] if isinstance(im_gen_adv, (list, tuple)) else im_gen_adv
            out_adv_anon = im_gen_adv_t * im_msk + x_final * (1.0 - im_msk)
            ea_final = self._embed_surrogate(out_adv_anon, crop_rect=face_crop_rect)

            # ---- (b) output baseline (pipeline pulita) ----
            im_lndm_base = self._render_landmarks_soft(pts_xy, torch.zeros_like(pts_xy), H, W)
            inp_base = torch.cat((im_lndm_base, x_res * (1.0 - im_msk)), dim=1)
            im_gen_base = self.netG(inp_base, onehot=labels_one_hot)
            im_gen_base_t = im_gen_base[0] if isinstance(im_gen_base, (list, tuple)) else im_gen_base
            out_base_anon = im_gen_base_t * im_msk + x_res * (1.0 - im_msk)
            e_base = self._embed_surrogate(out_base_anon, crop_rect=face_crop_rect)

            # ---- (c) calcolo similarità (tutte e tre) ----
            if ea_final is not None:
                final_sim = F.cosine_similarity(e0_ref, ea_final, dim=-1).item()
            if e_base is not None:
                baseline_orig_sim = F.cosine_similarity(e0_ref, e_base, dim=-1).item()
            if e_base is not None and ea_final is not None:
                baseline_sim = F.cosine_similarity(e_base, ea_final, dim=-1).item()

        self._last_oracle_sim = final_sim
        self._last_baseline_orig_sim = baseline_orig_sim
        self._last_baseline_sim = baseline_sim   # <-- FIX: ora viene valorizzato

        improvement = final_sim - baseline_orig_sim
        success = bool(
            ea_final is not None
            and e_base is not None
            and final_sim > SUCCESS_SIM_THRESHOLD
            and improvement > SUCCESS_IMPROVEMENT_MARGIN
        )

        # Log sicuro (nessuna variabile undefined)
        print(f"[CiaganLeakage] final_sim={final_sim:.4f} "
              f"baseline_orig_sim={baseline_orig_sim:.4f} "
              f"baseline_vs_adv={baseline_sim:.4f} "
              f"improvement={improvement:+.4f} success={success}")
        print(f"[DEBUG] e_base.shape={None if e_base is None else tuple(e_base.shape)} "
              f"ea_final.shape={None if ea_final is None else tuple(ea_final.shape)} "
              f"baseline_vs_adv_raw={baseline_sim!r}")

        return out_img.detach(), success, num_iter

    def attack(self, image_tensor: torch.Tensor, epsilon=None, **kwargs) -> torch.Tensor:
        if epsilon is None:
            epsilon = getattr(self, "epsilon", 32.0)
        img_adv, _, _ = self.perturb(image_tensor, epsilon=epsilon, **kwargs)
        return img_adv
