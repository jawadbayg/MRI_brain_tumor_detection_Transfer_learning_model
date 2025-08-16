# src/trainer.py
import json, os
import tensorflow as tf

def compile_and_train(model, train_ds, val_ds, epochs=5, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return history

def fine_tune(model, train_ds, val_ds, epochs=5, lr=1e-4, unfreeze_at=100):
    """
    Unfreezes top layers of the base model for light fine-tuning.
    """
    # Unfreeze last N layers of the backbone (MobileNetV2 has ~155 layers)
    trainable = False
    for i, layer in enumerate(model.layers):
        # Find base model
        if hasattr(layer, "layers"):  # nested model
            for j, l in enumerate(layer.layers):
                if j >= unfreeze_at:
                    l.trainable = True
                else:
                    l.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_accuracy"
        )
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    return history

def save_model_and_labels(model, out_path: str, class_names, labels_json="models/class_names.json"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)  # .keras format recommended
    with open(labels_json, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
