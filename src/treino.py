import numpy as np
import pandas as pd
import keras.callbacks, keras.backend, keras.utils
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import CURRENT_MODEL, EPOCHS, FOLDS, PATIENCE, SEED
from pipeline import criar_dataset
from modelo import criar_modelo


def treinar_um_fold(caminhos_treino, rotulos_treino, caminhos_validacao, rotulos_validacao, classe_unica, epochs = EPOCHS):
    dataset_treino = criar_dataset(caminhos_treino, rotulos_treino, classe_unica, shuffle = True)
    dataset_validacao = criar_dataset(caminhos_validacao, rotulos_validacao, classe_unica, shuffle = False)

    keras.backend.clear_session()
    modelo = criar_modelo(num_classes = len(classe_unica))

    inicio = time()

    historico = modelo.fit(
        dataset_treino,
        validation_data = dataset_validacao,
        epochs = epochs,
        verbose = 1,
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss", patience = PATIENCE, restore_best_weights = True
            )
        ]
    )

    tempo_total = time() - inicio

    probabilidade = modelo.predict(dataset_validacao, verbose = 0)
    classes_previstas = np.argmax(probabilidade, axis = 1)
    classes_reais = np.argmax(
        keras.utils.to_categorical(
            [np.where(classe_unica == rotulo)[0][0] for rotulo in rotulos_validacao],
            num_classes = len(classe_unica)
        ),

        axis = 1
    )

    val_loss, val_acc = modelo.evaluate(dataset_validacao, verbose = 0)

    return {
        "fold": None,
        "acuracia": accuracy_score(classes_reais, classes_previstas),
        "precisao_macro": precision_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "recall_macro": recall_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "f1_macro": f1_score(classes_reais, classes_previstas, average = "macro", zero_division = 0),
        "f1_weighted": f1_score(classes_reais, classes_previstas, average = "weighted", zero_division = 0),
        "loss": val_loss,
        "tempo_treino_seg": tempo_total,
        "epocas": len(historico.history["loss"])
    }


def rodar_kfold(train_val_paths, train_val_labels, classe_unica, folds = FOLDS):
    resultados_folds = []
    skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = SEED)
    train_val_paths = np.array(train_val_paths)
    train_val_labels = np.array(train_val_labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_paths, train_val_labels)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{folds}\n{'=' * 50}")

        resultado_folds = treinar_um_fold(
            train_val_paths[train_idx], train_val_labels[train_idx],
            train_val_paths[val_idx], train_val_labels[val_idx],
            classe_unica
        )

        resultado_folds["fold"] = fold + 1
        resultados_folds.append(resultado_folds)

        print(
            f"Acurácia: {resultado_folds['acuracia']:.4f} | "
            f"F1 macro: {resultado_folds['f1_macro']:.4f} | "
            f"Tempo: {resultado_folds['tempo_treino_seg']:.1f}s"
        )

    df_folds = pd.DataFrame(resultados_folds)

    resumo = {
        "modelo": CURRENT_MODEL,
        "acuracia_media": df_folds["acuracia"].mean(),
        "acuracia_std": df_folds["acuracia"].std(),
        "f1_macro_medio": df_folds["f1_macro"].mean(),
        "f1_macro_std": df_folds["f1_macro"].std(),
        "tempo_medio_seg": df_folds["tempo_treino_seg"].mean(),
        "epocas_medias": df_folds["epocas"].mean()
    }

    return df_folds, pd.DataFrame([resumo])