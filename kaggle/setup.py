import sys
import os
import subprocess

def setup_environment():
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    
    dependencies = [
        ("face-detection (DSFD)", "git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git"),
        ("tops (torch_ops)", "git+https://github.com/hukkelas/torch_ops.git")
    ]
    
    for name, url in dependencies:
        print(f"Installazione di {name}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", url, "--quiet"],
                env=env
            )
            print(f" Pacchetto {name} installato correttamente.")
        except Exception as e:
            print(f" Errore durante l'installazione di {name}: {e}")
            raise RuntimeError(f"Impossibile installare {name}.") from e

    print("Ambiente Kaggle configurato con successo.")

if __name__ == "__main__":
    setup_environment()
