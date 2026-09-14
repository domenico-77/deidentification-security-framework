import sys
import os
import subprocess

def setup_environment():
    print("Installazione del pacchetto 'tops' da hukkelas/torch_ops...")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    
    url = "git+https://github.com/hukkelas/torch_ops.git"
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", url, "--quiet"],
            env=env
        )
        print(" Pacchetto 'tops' (torch_ops) installato correttamente.")
    except Exception as e:
        print(f" Errore durante l'installazione di torch_ops: {e}")
        raise RuntimeError("Impossibile installare 'tops' da torch_ops.") from e

    print("Ambiente Kaggle configurato con successo.")

if __name__ == "__main__":
    setup_environment()
