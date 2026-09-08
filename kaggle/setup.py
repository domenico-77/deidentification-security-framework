import os
import subprocess

def setup_kaggle_environment():
    print("Installazione del pacchetto 'tops' da GitHub...")
    try:
        # Installa la libreria 'tops' richiesta da DeepPrivacy2
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/hukkelas/tops.git",
            "--quiet"
        ])
        print("Pacchetto 'tops' installato correttamente.")
    except Exception as e:
        print(f"Errore durante l'installazione di tops: {e}")

    print("Ambiente Kaggle configurato con successo.")

if __name__ == "__main__":
    setup_kaggle_environment()
