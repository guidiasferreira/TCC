import shutil
import numpy as np
import kagglehub
from pathlib import Path

from config_modelos import BASE_FOLDER_RGB, DATASET_KAGGLE

# Baixar dataset importado do Kaggle (PlantVillage)
def baixar_dataset():
    path = kagglehub.dataset_download(DATASET_KAGGLE)
    print(path)

    return Path(path)


# Copiar as pastas que contém imagens da cultura do tomate em formato RGB
def copiar_pastas(origem):
    destino = BASE_FOLDER_RGB 
    destino.mkdir(parents = True, exist_ok = True)

    pastas_copiadas = 0

    for pasta in origem.rglob("*Tomato*"):
        if (pasta.is_dir() and "color" in pasta.parts):
            caminho_destino = destino / pasta.name
            
            if (not caminho_destino.exists()):
                shutil.copytree(pasta, caminho_destino)
                pastas_copiadas += 1

    return destino


# Coletar os caminhos (Paths) e rótulos (Labels)
def coletar_caminhos_rotulos(pasta_destino):
    paths = []
    labels = []

    extensoes = ("*.jpg", "*.jpeg", "*.png")

    for classe_pasta in sorted(pasta_destino.iterdir()):
        if (classe_pasta.is_dir()):
            for extensao in extensoes:
                for img in sorted(classe_pasta.glob(extensao)):
                    paths.append(str(img))
                    labels.append(classe_pasta.name)

    paths = np.array(paths)
    labels = np.array(labels)

    total_imagens = len(paths)

    print(f"\nTotal de imagens encontradas: {total_imagens}")
    
    classes_unicas, contagens = np.unique(labels, return_counts = True)

    print("-" * 50)
    print("Distribuição das classes:")
    for classe, contagem in zip(classes_unicas, contagens):
        percentual = (contagem / total_imagens) * 100
        print(f"{classe}: {contagem} imagens ({percentual:.2f}%)")
    print("-" * 50)
    print(f"Quantidade de classes: {len(classes_unicas)}")

    return paths, labels