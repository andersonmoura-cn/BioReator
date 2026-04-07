from config.config import imgs
import matplotlib.pyplot as plt
from pathlib import Path


def img_save(file_name, dir_name, dpi = 300, show: bool = False):
    folder = imgs / dir_name
    folder.mkdir(parents=True, exist_ok=True)

    # Caminho completo do arquivo
    file_path = folder / f"{file_name}.png"

    plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    
    plt.close()

    print(f"Imagem salva em: {file_path}")