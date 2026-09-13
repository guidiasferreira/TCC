import kagglehub
import shutil
import tensorflow as tf
import numpy as np
import pandas as pd
import keras.layers, keras.utils, keras.backend, keras.optimizers, keras.callbacks, keras.applications.efficientnet, keras.applications.mobilenet_v2, keras.applications.mobilenet_v3, keras.applications.resnet_v2, keras.applications.densenet, keras.applications.nasnet
from keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small, NASNetMobile, EfficientNetB2, EfficientNetB3, ResNet50V2, DenseNet121
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from time import time

BATCH_SIZE = 32
EPOCHS = 100
IMG_SIZE = (112, 112)
FOLDS = 5

BASE_FOLDER_GRAYSCALE = Path("../data/images_grayscale")
BASE_FOLDER_RGB = Path("../data/images_rgb")

MODELO_ATUAL = "MobileNetV3Large"
MODELOS = {
    "MobileNetV2": {
        "classe": MobileNetV2,
        "preprocess": keras.applications.mobilenet_v2.preprocess_input
    },

    "MobileNetV3Large": {
        "classe": MobileNetV3Large,
        "preprocess": keras.applications.mobilenet_v3.preprocess_input
    },

    "MobileNetV3Small": {
        "classe": MobileNetV3Small,
        "preprocess": keras.applications.mobilenet_v3.preprocess_input
    },

    "EfficientNetB2": {
        "classe": EfficientNetB2,
        "preprocess": keras.applications.efficientnet.preprocess_input
    },

    "EfficientNetB3": {
        "classe": EfficientNetB3,
        "preprocess": keras.applications.efficientnet.preprocess_input
    },

    "ResNet50V2": {
        "classe": ResNet50V2,
        "preprocess": keras.applications.resnet_v2.preprocess_input
    },

    "DenseNet121": {
        "classe": DenseNet121,
        "preprocess": keras.applications.densenet.preprocess_input
    },

    "NASNetMobile": {
        "classe": NASNetMobile,
        "preprocess": keras.applications.nasnet.preprocess_input
    }
}


def baixar_dataset():
    path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    print(path)

    return Path(path)


def copiar_pastas_rgb(origem):
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
        test_size = 0.15,
        stratify = labels,
        random_state = 42
    )

    print(f"Treino + Validação (K-Fold): {len(train_val_paths)} imagens")
    print(f"Teste (guardado até o final): {len(test_paths)} imagens")

    return train_val_paths, test_paths, train_val_labels, test_labels


def carregar_processar_imagem(path, size = IMG_SIZE):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.image.resize(image, size)
    image = MODELOS[MODELO_ATUAL]["preprocess"](image)

    return image


def criar_dataset(list_path, list_label, unique_class, batch_size = BATCH_SIZE, size = IMG_SIZE, shuffle = True):
    indices_rotulos = [np.where(unique_class == r)[0][0] for r in list_label]
    rotulos_one_hot = keras.utils.to_categorical(indices_rotulos, num_classes = len(unique_class))
    
    dataset = tf.data.Dataset.from_tensor_slices((list_path, rotulos_one_hot))
    
    if (shuffle):
        dataset = dataset.shuffle(buffer_size = len(list_path), seed = 42)
    
    dataset = dataset.map(
        lambda caminho, rotulo: (carregar_processar_imagem(caminho, size), rotulo),
        num_parallel_calls = tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def criar_modelo(num_classes):
    classe_modelo = MODELOS[MODELO_ATUAL]["classe"]

    base_model = classe_modelo(
        include_top = False,
        weights = "imagenet",
        input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(512, activation = "relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation = "softmax")
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 0.001),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model


def treinar_um_fold(caminhos_treino, rotulos_treino, caminhos_val, rotulos_val, unique_class, epochs = EPOCHS):
    dataset_treino = criar_dataset(caminhos_treino, rotulos_treino, unique_class, shuffle = True)
    dataset_val = criar_dataset(caminhos_val, rotulos_val, unique_class, shuffle = False)

    keras.backend.clear_session()
    modelo = criar_modelo(num_classes = len(unique_class))

    inicio = time()

    modelo.fit(
        dataset_treino,
        validation_data = dataset_val,
        epochs = epochs,
        verbose = 1,
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor = "val_loss", patience = 3, restore_best_weights = True
            )
        ]
    )

    tempo_total = time() - inicio

    val_loss, val_acc = modelo.evaluate(dataset_val, verbose = 0)

    return val_acc, val_loss, tempo_total


def rodar_kfold(train_val_paths, train_val_labels, unique_class, folds = FOLDS, arquivo_saida = f"../results/{MODELO_ATUAL}/resultados_{MODELO_ATUAL.lower()}_rgb_{IMG_SIZE[0]}x{IMG_SIZE[1]}v2.csv"):
    acuracias = []
    loss = []
    tempos = []

    skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 42)
    train_val_paths = np.array(train_val_paths)
    train_val_labels = np.array(train_val_labels)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_paths, train_val_labels)):
        print(f"\n{'=' * 50}\nFold {fold + 1}/{folds}\n{'=' * 50}")

        val_acc, val_loss, tempo_total = treinar_um_fold(
            train_val_paths[train_idx], train_val_labels[train_idx],
            train_val_paths[val_idx], train_val_labels[val_idx],
            unique_class
        )

        acuracias.append(val_acc)
        loss.append(val_loss)
        tempos.append(tempo_total)

        print(f"Acurácia: {val_acc:.4f} | Tempo: {tempo_total:.1f}s")

    resultados = {
        "modelo": [MODELO_ATUAL],
        "acuracia_media": [np.mean(acuracias)],
        "tempo_medio_seg": [np.mean(tempos)]
    }

    caminho_csv = Path(arquivo_saida)
    caminho_csv.parent.mkdir(parents = True, exist_ok = True)
    df_resultado = pd.DataFrame(resultados)
    df_resultado.to_csv(caminho_csv, index = False)

    print(f"\nK-Fold completo. Resultados salvos em {arquivo_saida}")
    return df_resultado


def resumir_resultados(df_resultados):
    print(f"\nModelo: {df_resultados['modelo'][0]}")
    print(f"Acurácia média: {df_resultados['acuracia_media'][0]:.4f}")
    print(f"Tempo médio por fold: {df_resultados['tempo_medio_seg'][0]:.1f}s")

    return df_resultados


if (__name__ == "__main__"):
    dataset_path = baixar_dataset()
    pasta_destino = copiar_pastas_rgb(dataset_path)
    paths, labels = coletar_caminhos_rotulos(pasta_destino)
    train_val_paths, test_paths, train_val_labels, test_labels = separar_test_set(paths, labels)

    unique_class = np.unique(labels)
    df_resultado = rodar_kfold(train_val_paths, train_val_labels, unique_class)

    resumir_resultados(df_resultado)