# src/model_mobilenet.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def build_transfer_model(img_size=(224, 224), num_classes=4, train_base=False):
    """
    MobileNetV2 transfer model with a small custom head.
    `train_base=False` for warmup; set True later for fine-tuning top layers.
    """
    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg"
    )
    base.trainable = train_base  # freeze for warmup

    # Small head
    y = layers.Dropout(0.25)(base.output)
    y = layers.Dense(256, activation="relu")(y)
    y = layers.Dropout(0.25)(y)
    outputs = layers.Dense(num_classes, activation="softmax")(y)

    model = Model(inputs, outputs)
    return model
