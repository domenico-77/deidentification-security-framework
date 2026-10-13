import sys
import os
from pathlib import Path
import torch
import numpy as np
from torchvision import transforms

KAGGLE_INPUT = Path("/kaggle/input")

if KAGGLE_INPUT.exists():
    repo_matches = list(KAGGLE_INPUT.glob("**/ciagan-repository"))
    models_matches = list(KAGGLE_INPUT.glob("**/ciagan-models"))
    
    if repo_matches:
        ciagan_repo_root = (repo_matches[0] / "ciagan").resolve()
    else:
        ciagan_repo_root = Path("/kaggle/input/ciagan-repository/ciagan").resolve()
        
    if models_matches:
        ciagan_models_dir = models_matches[0].resolve()
    else:
        ciagan_models_dir = Path("/kaggle/input/ciagan-models").resolve()
else:
    base_local = Path("./data")
    ciagan_repo_root = (base_local / "ciagan-repository/ciagan").resolve()
    ciagan_models_dir = (base_local / "ciagan-models").resolve()

ciagan_source_dir = ciagan_repo_root / "source"
dataset_path = ciagan_repo_root / "dataset" / "celeba"

for p in [ciagan_repo_root, ciagan_source_dir]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from targets.base_target import BaseDeidentificationTarget
import util_data

class CIAGANTarget(BaseDeidentificationTarget):
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.dataset = self._load_dataset()

    def _load_model(self):
        from arch.arch_unet_flex import Generator
        model = Generator()
        
        checkpoint_path = ciagan_models_dir / "modelG_ciagan.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"[ERRORE] Checkpoint non trovato in: {checkpoint_path}")

        print(f"[INFO] Caricamento pesi CIAGAN da: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        
        if isinstance(checkpoint, dict):
            for key in ["state_dict", "model_state_dict", "generator", "G"]:
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

        model.load_state_dict(checkpoint, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _load_dataset(self):
        num_folders = len([d for d in os.listdir(dataset_path / 'lndm') if (dataset_path / 'lndm' / d).is_dir()])
        dataset = util_data.ImageDataset(
            root_dir=str(dataset_path) + "/",
            label_num=num_folders,
            transform_fnc=transforms.Compose([transforms.ToTensor()]),
            flag_sample=1,
            flag_augment=False
        )
        return dataset

    def process_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        device = self.device
        
        sample = self.dataset[0]
        im_faces, im_lndm, im_msk, im_ind = sample
        
        # Gestione nel caso in cui siano restituiti come liste
        if isinstance(im_faces, list):
            im_faces = im_faces[0]
        if isinstance(im_lndm, list):
            im_lndm = im_lndm[0]
        if isinstance(im_msk, list):
            im_msk = im_msk[0]
        if isinstance(im_ind, list):
            im_ind = im_ind[0]

        # Conversione in tensori se sono immagini PIL
        to_tensor = transforms.ToTensor()
        if not isinstance(im_faces, torch.Tensor):
            im_faces = to_tensor(im_faces)
        if not isinstance(im_lndm, torch.Tensor):
            im_lndm = to_tensor(im_lndm)
        if not isinstance(im_msk, torch.Tensor):
            im_msk = to_tensor(im_msk)

        im_faces = im_faces.float().unsqueeze(0).to(device)
        im_lndm = im_lndm.float().unsqueeze(0).to(device)
        im_msk = im_msk.float().unsqueeze(0).to(device)
        
        if not isinstance(im_ind, torch.Tensor):
            im_ind = torch.tensor([im_ind], dtype=torch.long)
        else:
            im_ind = im_ind.long().view(1)

        output_id = (int(im_ind.cpu().item()) + 1) % len(self.dataset)
        labels_one_hot = np.zeros((1, 1200))
        labels_one_hot[0, output_id % 1200] = 1
        labels_one_hot = torch.tensor(labels_one_hot).float().to(device)

        with torch.no_grad():
            input_gen = torch.cat((im_lndm, im_faces * (1 - im_msk)), dim=1)
            
            im_gen = self.model(input_gen, onehot=labels_one_hot)
            im_gen = torch.clamp(im_gen * im_msk + im_faces * (1 - im_msk), 0, 1)

        output_tensor = im_gen[0]
        return output_tensor.to(device)
