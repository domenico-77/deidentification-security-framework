import os

def setup_kaggle_environment():
    """Installa le dipendenze e configura la variabile PYTHONPATH per Kaggle."""
    os.system("pip install -q hydra-core motpy torch-fidelity torchattacks webdataset")
    os.system("pip install -q 'git+https://github.com/facebookresearch/detectron2.git'")
    print("Ambiente Kaggle configurato con successo.")

if __name__ == "__main__":
    setup_kaggle_environment()