import keras.applications.mobilenet_v3, keras.applications.efficientnet_v2, keras.applications.resnet_v2, keras.applications.densenet
from keras.applications import MobileNetV3Large, MobileNetV3Small, EfficientNetV2B3, EfficientNetV2B0, ResNet50V2, DenseNet121
from pathlib import Path

# Parâmetros que serão utilizados para treinar o modelo
BATCH_SIZE = 32
EPOCHS = 200
IMG_SIZE = (224, 224)
FOLDS = 5
SEED = 42
PATIENCE = 15

# Pastas e arquivos
DATASET_KAGGLE = "abdallahalidev/plantvillage-dataset"
BASE_FOLDER_GRAYSCALE = Path("data/images_grayscale")
BASE_FOLDER_RGB = Path("data/images_rgb")
MODEL_FOLDER = Path("results/models")
EVALUATION_FOLDER = Path("results/evaluation_v3")

# Modelos pré-treinados
MODELS = {
    "MobileNetV3Large": {
        "classe": MobileNetV3Large,
        "preprocess": keras.applications.mobilenet_v3.preprocess_input
    },

    "MobileNetV3Small": {
        "classe": MobileNetV3Small,
        "preprocess": keras.applications.mobilenet_v3.preprocess_input
    },

    "EfficientNetV2B3": {
        "classe": EfficientNetV2B3,
        "preprocess": keras.applications.efficientnet_v2.preprocess_input
    },

    "EfficientNetV2B0": {
        "classe": EfficientNetV2B0,
        "preprocess": keras.applications.efficientnet_v2.preprocess_input
    },

    "ResNet50V2": {
        "classe": ResNet50V2,
        "preprocess": keras.applications.resnet_v2.preprocess_input
    },

    "DenseNet121": {
        "classe": DenseNet121,
        "preprocess": keras.applications.densenet.preprocess_input
    },
}

# Escolha do modelo atual
CURRENT_MODEL = "MobileNetV3Large"