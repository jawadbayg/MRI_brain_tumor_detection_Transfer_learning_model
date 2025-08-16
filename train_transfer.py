# train_transfer.py
from data_loader import load_datasets
from model_mobilenet import build_transfer_model
from trainer import compile_and_train, fine_tune, save_model_and_labels
from evaluator import evaluate

TRAIN_DIR = "dataset/Training"
TEST_DIR  = "dataset/Testing"
IMG_SIZE  = (224, 224)
BATCH     = 32

if __name__ == "__main__":
    # 1) data
    train_ds, val_ds, test_ds, class_names = load_datasets(
        TRAIN_DIR, TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH, val_split=0.2
    )
    print("Classes (training order):", class_names)

    # 2) warmup with frozen base
    model = build_transfer_model(img_size=IMG_SIZE, num_classes=len(class_names), train_base=False)
    compile_and_train(model, train_ds, val_ds, epochs=5, lr=1e-3)

    # 3) light fine-tuning (unfreeze top layers)
    history_ft = fine_tune(model, train_ds, val_ds, epochs=5, lr=1e-4, unfreeze_at=100)

    # 4) evaluate
    evaluate(model, test_ds, class_names)

    # 5) save model + label order
    save_model_and_labels(model, "models/brain_tumor_mobilenetv2.keras", class_names)
    print("Saved to models/brain_tumor_mobilenetv2.keras and models/class_names.json")
