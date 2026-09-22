import os
import numpy as np
import keras.utils
from sklearn.model_selection import StratifiedKFold

from config_modelos import SEED, BASE_FOLDER_RGB
from dados import coletar_caminhos_rotulos

os.environ["PYTHONHASHSEED"] = str(SEED)

keras.utils.set_random_seed(SEED)

def verificar_folds():
    print("Coletando imagens do disco (usando sorted)...")
    paths, labels = coletar_caminhos_rotulos(BASE_FOLDER_RGB)
    classe_unica = np.unique(labels)
    
    print("\nIniciando Divisão do K-Fold (SEED = 42)...")
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)

    cont = 5
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels)):
        while (fold <= cont):
            print(f"\n--- FOLD {fold + 1}: PROVA DE REPRODUTIBILIDADE ---")
            print(f"As primeiras 5 imagens exatas que entraram para VALIDAÇÃO no Fold {fold + 1} são: ")

            for i in range(5):
                indice = val_idx[i]
                caminho_imagem = paths[indice]
                print(f"{i + 1}. {caminho_imagem}")

            fold += 1

            break