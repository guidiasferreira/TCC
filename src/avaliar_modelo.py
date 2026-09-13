import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras.callbacks, keras.backend
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

from config import CURRENT_MODEL, EPOCHS, PATIENCE, SEED, TEST_SIZE, MODEL_FOLDER, EVALUATION_FOLDER, IMG_SIZE
from dados import train_test_split
from pipeline import criar_dataset
from modelo import criar_modelo


def treinar_modelo_final(train_val_paths, train_val_labels, classe_unica):
    caminhos_treino, caminhos_validacao, rotulos_treino, rotulos_validacao = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size = TEST_SIZE,
        stratify = train_val_labels,
        random_state = SEED
    )

    dataset_treino = criar_dataset(caminhos_treino, rotulos_treino, classe_unica, shuffle = True)
    dataset_validacao = criar_dataset(caminhos_validacao, rotulos_validacao, classe_unica, shuffle = False)

    keras.backend.clear_session()

    modelo = criar_modelo(num_classes = len(classe_unica))
    inicio = time()
    historico = modelo.fit(
        dataset_treino,
        validation_data = dataset_validacao,
        epochs = EPOCHS,
        verbose = 1,
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss", patience = PATIENCE, restore_best_weights = True
            )
        ]
    )

    tempo_total = time() - inicio

    MODEL_FOLDER.mkdir(parents = True, exist_ok = True)
    caminho_modelo = MODEL_FOLDER / f"{CURRENT_MODEL.lower()}_rgb_{IMG_SIZE[0]}x{IMG_SIZE[1]}_v3.keras"
    modelo.save(caminho_modelo)
    print(f"\nModelo final salvo em: {caminho_modelo}")

    return modelo, tempo_total, len(historico.history["loss"])


def avaliar_modelo(modelo, test_paths, test_labels, classe_unica):
    dataset_teste = criar_dataset(test_paths, test_labels, classe_unica, shuffle = False)
    inicio = time()
    probabilidades = modelo.predict(dataset_teste, verbose = 0)
    tempo_inferencia = time() - inicio
    classes_previstas = np.argmax(probabilidades, axis = 1)
    classes_reais = np.array([np.where(classe_unica == rotulo)[0][0] for rotulo in test_labels])

    metricas = {
        "modelo": CURRENT_MODEL,
        "acuracia": accuracy_score(classes_reais, classes_previstas),
        "precisao_macro": precision_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "recall_macro": recall_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "f1_macro": f1_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "f1_weighted": f1_score(classes_reais, classes_previstas, average = "weighted", zero_division = 0),
        "tempo_inferencia_seg": tempo_inferencia,
        "tempo_medio_por_imagem_seg": tempo_inferencia / len(test_paths)
    }
    relatorio = pd.DataFrame(
        classification_report(
            classes_reais,
            classes_previstas,
            labels = range(len(classe_unica)),
            target_names = classe_unica,
            output_dict = True,
            zero_division = 0
        )
    ).transpose().reset_index().rename(columns = {"index": "classe"})

    matriz = pd.DataFrame(
        confusion_matrix(classes_reais, classes_previstas, labels = range(len(classe_unica))),
        index = classe_unica,
        columns = classe_unica
    )

    EVALUATION_FOLDER.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (12, 10))
    sns.heatmap(
        matriz,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        cbar = True
    )

    plt.title(f"Matriz de Confusão - {CURRENT_MODEL}")
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.xticks(rotation = 45, ha =  "right")
    plt.yticks(rotation = 0)
    plt.tight_layout()

    caminho_matriz = (EVALUATION_FOLDER / "matrizes" / f"matriz_confusao_{CURRENT_MODEL.lower()}.png")
    caminho_matriz.parent.mkdir(parents = True, exist_ok = True)
    
    plt.savefig(caminho_matriz, dpi = 300)
    plt.close()

    print("\nAvaliação no conjunto de teste:")
    for nome, valor in metricas.items():
        if (nome not in {"modelo"}):
            print(f"{nome}: {valor:.4f}")

    print("\nRelatório por classe:")
    print(relatorio.to_string(index = False))

    return pd.DataFrame([metricas]), relatorio, matriz