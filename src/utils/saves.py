import matplotlib.pyplot as plt
from utils.tratamento import tf_to_num_den
from config.config import imgs

# registrar Função de transferencia em config
def save_mimo_tf(G, var, filename="config/tf_config.py"):
    outputs = ["X", "S"]
    inputs = ["D", "Sf"]

    with open(filename, "w") as f:
        # f.write("# Funcoes de transferencia MIMO\n\n")
        
        for i, y in enumerate(outputs):
            for j, u in enumerate(inputs):
                
                num, den = tf_to_num_den(G[i, j], var)
                
                name = f"G_{y}_{u}"
                
                f.write(f"{name} = {{\n")
                f.write(f"    'num': {num},\n")
                f.write(f"    'den': {den}\n")
                f.write("}\n\n")


# salvar imagens em imgs
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