# src/data_loader.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def load_datasets(
    train_dir: str,
    test_dir: str,
    img_size=(224, 224),
    batch_size=32,
    val_split=0.2,
    seed=1337,
):
    """
    Creates (train, val, test) tf.data.Datasets from directory structure.
    Returns datasets + class_names list in the *training* order.
    """
    train_ds = image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    test_ds = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        label_mode="categorical",
    )

    class_names = train_ds.class_names  # folder order used for training
    # Performance: cache+prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
