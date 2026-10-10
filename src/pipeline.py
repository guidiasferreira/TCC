import numpy as np
import tensorflow as tf
import keras.utils

from config_modelos import MODELS, CURRENT_MODEL, IMG_SIZE, BATCH_SIZE, SEED

# Carregar as imagens e realizar o pré-processamento, convertendo as imagens de 256x256 para 224x224
def carregar_processar_imagem(path, size = IMG_SIZE):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels = 3, expand_animations = False)
    image = tf.image.resize(image, size)
    image = MODELS[CURRENT_MODEL]["preprocess"](image)

    return image


# Cria o dataset
def criar_dataset(list_path, list_label, classe_unica, batch_size = BATCH_SIZE, size = IMG_SIZE, shuffle = True):
    indices_rotulos = [np.where(classe_unica == r)[0][0] for r in list_label]
    rotulos_one_hot = keras.utils.to_categorical(indices_rotulos, num_classes = len(classe_unica))
    
    dataset = tf.data.Dataset.from_tensor_slices((list_path, rotulos_one_hot))
    
    if (shuffle):
        dataset = dataset.shuffle(buffer_size = len(list_path), seed = SEED)
    
    dataset = dataset.map(
        lambda caminho, rotulo: (carregar_processar_imagem(caminho, size), rotulo),
        num_parallel_calls = tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset