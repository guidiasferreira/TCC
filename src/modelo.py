import keras.layers, keras.optimizers
from config import MODELS, CURRENT_MODEL, IMG_SIZE


def criar_modelo(num_classes):
    classe_modelo = MODELS[CURRENT_MODEL]["classe"]

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
        optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model