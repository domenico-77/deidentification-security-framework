import sys
import os
import subprocess

def setup_environment():
    print("Installazione delle dipendenze di DeepPrivacy2 (tops)...")
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    
    # URL ufficiali della libreria tops (torch-build) di Håkon Hukkelås
    sources = [
        "git+https://github.com/hukkelas/torch-build.git",
        "git+https://github.com/hukkelas/torch_build.git",
        "tops"
    ]
    
    installed = False
    for src in sources:
        try:
            print(f"Tentativo installazione da: {src}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", src, "--quiet"],
                env=env
            )
            print(f" Installazione riuscita da: {src}")
            installed = True
            break
        except Exception as e:
            print(f" Fallito: {src}")

    if not installed:
        raise RuntimeError("Impossibile installare la libreria 'tops'. Verificare la connessione o l'URL del repository.")

    print("Ambiente Kaggle configurato con successo.")

if __name__ == "__main__":
    setup_environment()
