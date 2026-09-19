"""Train a transfer-learning ResNet-50 classifier on the FYP dataset."""
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
import tensorflow as tf

CLASSES = ["Normal", "immature", "mature"]


class TrainingHistoryRecorder(tf.keras.callbacks.Callback):
    """Save loss and accuracy for every epoch in a reusable FYP record."""

    def __init__(self, csv_path, json_path, run_id, phase):
        super().__init__()
        self.csv_path = csv_path
        self.json_path = json_path
        self.run_id = run_id
        self.phase = phase

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {
            "run_id": self.run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0)),
            "accuracy": float(logs.get("accuracy", 0)),
            "val_loss": float(logs.get("val_loss", 0)),
            "val_accuracy": float(logs.get("val_accuracy", 0)),
        }
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        records = []
        if self.json_path.exists():
            records = json.loads(self.json_path.read_text(encoding="utf-8"))
        records.append(row)
        self.json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("train_%Y%m%dT%H%M%SZ")
    history_csv = models_dir / "training_history.csv"
    history_json = models_dir / "training_history.json"
    train_ds = tf.keras.utils.image_dataset_from_directory(args.data_dir, labels="inferred", label_mode="categorical", validation_split=0.2, subset="training", seed=42, image_size=(224, 224), batch_size=args.batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(args.data_dir, labels="inferred", label_mode="categorical", validation_split=0.2, subset="validation", seed=42, image_size=(224, 224), batch_size=args.batch_size)
    print("Detected classes:", train_ds.class_names)
    augment = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.04), tf.keras.layers.RandomZoom(0.1), tf.keras.layers.RandomContrast(0.12)])
    base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    x = augment(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        TrainingHistoryRecorder(history_csv, history_json, run_id, "feature_extraction"),
    ]
    model.fit(train_ds.prefetch(tf.data.AUTOTUNE), validation_data=val_ds.prefetch(tf.data.AUTOTUNE), epochs=args.epochs, callbacks=callbacks)
    base.trainable = True
    for layer in base.layers[:-20]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        TrainingHistoryRecorder(history_csv, history_json, run_id, "fine_tuning"),
    ]
    model.fit(train_ds.prefetch(tf.data.AUTOTUNE), validation_data=val_ds.prefetch(tf.data.AUTOTUNE), epochs=10, callbacks=fine_tune_callbacks)
    output = models_dir / "cataract_resnet50.keras"
    model.save(output)
    print(f"Saved trained model to {output}")
    print(f"Saved training metrics to {history_csv}")


if __name__ == "__main__": main()
