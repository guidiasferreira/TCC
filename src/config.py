import keras.applications.efficientnet, keras.applications.mobilenet_v2, keras.applications.mobilenet_v3, keras.applications.resnet_v2, keras.applications.densenet, keras.applications.nasnet
from keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small, NASNetMobile, EfficientNetB2, EfficientNetB3, ResNet50V2, DenseNet121
from pathlib import Path

BATCH_SIZE = 32
EPOCHS = 50
IMG_SIZE = (160, 160)
FOLDS = 5
TEST_SIZE = 0.20
SEED = 42
PATIENCE = 5

DATASET_KAGGLE = "abdallahalidev/plantvillage-dataset"
BASE_FOLDER_GRAYSCALE = Path("data/images_grayscale")
BASE_FOLDER_RGB = Path("data/images_rgb")
MODEL_FOLDER = Path("results/models")
EVALUATION_FOLDER = Path("results/evaluation_v2")

MODELS = {
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

CURRENT_MODEL = "MobileNetV2"