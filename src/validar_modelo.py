import numpy as np
import pandas as pd
import keras.callbacks, keras.backend, keras.utils
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from time import time

from config_modelos import CURRENT_MODEL, EPOCHS, FOLDS, PATIENCE, SEED
from pipeline import criar_dataset
from modelo import criar_modelo
from relatorios import gerar_matriz_confusao


def treinar_fold(caminhos_treino, rotulos_treino, caminhos_validacao, rotulos_validacao, classe_unica, epochs = EPOCHS):
    keras.backend.clear_session()

    dataset_treino = criar_dataset(caminhos_treino, rotulos_treino, classe_unica, shuffle = True)
    dataset_validacao = criar_dataset(caminhos_validacao, rotulos_validacao, classe_unica, shuffle = False)

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
        "epocas": len(historico.history["loss"]),
        "matriz_confusao": confusion_matrix(classes_reais, classes_previstas)
    }


def rodar_kfold(train_val_paths, train_val_labels, classe_unica, folds = FOLDS):
    resultados_folds = []
    matriz_confusao_global = np.zeros((len(classe_unica), len(classe_unica)), dtype = int)

    skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = SEED)
    
    train_val_paths = np.array(train_val_paths)
    train_val_labels = np.array(train_val_labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_paths, train_val_labels)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{folds}\n{'=' * 50}")
        print(f"Imagens de Treino: {len(train_idx)} | Imagens de Validação: {len(val_idx)}")
        print(f"{'=' * 50}")

        resultado_folds = treinar_fold(
            train_val_paths[train_idx], train_val_labels[train_idx],
            train_val_paths[val_idx], train_val_labels[val_idx],
            classe_unica
        )

        resultado_folds["fold"] = fold + 1
        resultados_folds.append(resultado_folds)

        matriz_confusao_global += resultado_folds["matriz_confusao"]
        del resultado_folds["matriz_confusao"]

        print(f"Acurácia: {resultado_folds['acuracia']:.4f} | Loss: {resultado_folds['loss']:.4f} | Tempo: {resultado_folds['tempo_treino_seg']:.2f}s")

    df_folds = pd.DataFrame(resultados_folds)

    resumo = {
        "modelo": CURRENT_MODEL,
        "acuracia_media": df_folds["acuracia"].mean(),
        "acuracia_std": df_folds["acuracia"].std(),
        "precisao_macro_medio": df_folds["precisao_macro"].mean(),
        "precisao_macro_std": df_folds["precisao_macro"].std(),
        "recall_macro_medio": df_folds["recall_macro"].mean(),
        "recall_macro_std": df_folds["recall_macro"].std(),
        "f1_macro_medio": df_folds["f1_macro"].mean(),
        "f1_macro_std": df_folds["f1_macro"].std(),
        "tempo_medio_seg": df_folds["tempo_treino_seg"].mean(),
        "epocas_medias": df_folds["epocas"].mean()
    }

    gerar_matriz_confusao(matriz_confusao_global, classe_unica)

    return df_folds, pd.DataFrame([resumo])