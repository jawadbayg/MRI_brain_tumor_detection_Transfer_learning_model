# src/evaluator.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_ds, class_names):
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # Build predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    y_true_idx = np.argmax(y_true, axis=1)

    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred_idx = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true_idx, y_pred_idx, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true_idx, y_pred_idx))
