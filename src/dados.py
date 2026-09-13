import shutil
import numpy as np
import kagglehub
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import BASE_FOLDER_RGB, DATASET_KAGGLE, TEST_SIZE, SEED


def baixar_dataset():
    path = kagglehub.dataset_download(DATASET_KAGGLE)
    print(path)

    return Path(path)


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


def coletar_caminhos_rotulos(pasta_destino):
    paths = []
    labels = []

    for classe_pasta in pasta_destino.iterdir():
        if (classe_pasta.is_dir()):
            for img in classe_pasta.glob("*.JPG"):
                paths.append(str(img))
                labels.append(classe_pasta.name)

    paths = np.array(paths)
    labels = np.array(labels)

    print(f"Total de imagens encontradas: {len(paths)}")

    return paths, labels


def separar_test_set(paths, labels):
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size = TEST_SIZE,
        stratify = labels,
        random_state = SEED
    )

    print(f"Treino + Validação (K-Fold): {len(train_val_paths)} imagens")
    print(f"Teste (guardado até o final): {len(test_paths)} imagens")

    return train_val_paths, test_paths, train_val_labels, test_labels