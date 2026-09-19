"""Transparent local evaluation for the trained cataract classifier."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .model_service import CLASSES, DATASET_PATH, MODEL_PATH, load_model, preprocess

RESULT_PATH = MODEL_PATH.parent / "evaluation_result.json"


def _safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def evaluate_model(samples_per_class=25):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Train the model before running evaluation.")
    if not DATASET_PATH.exists():
        raise FileNotFoundError("The dataset folder is unavailable for evaluation.")

    images, true_labels = [], []
    folder_to_index = {"normal": 0, "immature": 1, "mature": 2}
    for folder in sorted(path for path in DATASET_PATH.iterdir() if path.is_dir()):
        class_index = folder_to_index.get(folder.name.lower())
        if class_index is None:
            continue
        files = sorted(path for path in folder.iterdir() if path.is_file())
        if not files:
            continue
        # Spread selections across each folder instead of taking consecutive files.
        positions = np.linspace(0, len(files) - 1, min(samples_per_class, len(files)), dtype=int)
        for position in positions:
            try:
                _, image = preprocess(files[int(position)].read_bytes())
                images.append(image)
                true_labels.append(class_index)
            except Exception:
                continue
    if len(images) < 9:
        raise ValueError("Not enough readable dataset images were available for evaluation.")

    probabilities = load_model().predict(np.stack(images), verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    targets = np.asarray(true_labels)
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for actual, predicted in zip(targets, predictions):
        matrix[actual, predicted] += 1

    per_class = []
    f1_scores = []
    for index, label in enumerate(CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_scores.append(f1)
        per_class.append({"class": label, "precision": precision, "recall": recall, "f1": f1, "support": support})

    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": int(len(targets)),
        "samples_per_class_requested": samples_per_class,
        "accuracy": float(np.mean(predictions == targets)),
        "macro_f1": float(np.mean(f1_scores)),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "classes": CLASSES,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "notice": "This is a reproducible internal dataset sample evaluation. It is not external clinical validation and should not be presented as clinical performance.",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def last_evaluation():
    if not RESULT_PATH.exists():
        return None
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
