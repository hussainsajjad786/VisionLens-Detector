from pathlib import Path
from functools import lru_cache
import io
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "cataract_resnet50_previous.keras"
REFERENCE_PATH = ROOT / "models" / "eye_reference_previous.npz"
DATASET_PATH = ROOT / "data" / "raw" / "archive" / "train"
CLASSES = ["Normal", "Immature", "Mature"]


def preprocess(raw: bytes):
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    original = np.array(image)
    resized = cv2.resize(original, (224, 224))
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    filtered = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return original, filtered.astype("float32")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


def _embedding_model(model):
    """Return the trained ResNet feature extractor before the classification head."""
    import tensorflow as tf
    return tf.keras.Model(model.inputs, model.get_layer("global_average_pooling2d").output)


def _normalise(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def _make_eye_reference(model):
    """Create a compact reference profile from real eyes in the FYP dataset."""
    samples = []
    for folder in sorted(DATASET_PATH.iterdir()):
        if folder.is_dir():
            for path in sorted(folder.glob("*"))[:25]:
                if path.is_file():
                    try:
                        _, image = preprocess(path.read_bytes())
                        samples.append(image)
                    except Exception:
                        continue
    if len(samples) < 20:
        raise ValueError("Eye-image validation cannot start because the reference dataset is unavailable.")
    vectors = _normalise(_embedding_model(model).predict(np.stack(samples), verbose=0))
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -1)
    threshold = 0.38
    np.savez_compressed(REFERENCE_PATH, vectors=vectors, threshold=threshold)
    return vectors, threshold


@lru_cache(maxsize=1)
def _eye_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    if detector.empty():
        raise RuntimeError("The eye-image validator could not be loaded.")
    return detector


def _contains_large_eye(original):
    """Detect a prominent eye, not a tiny eye-like pattern in an object image."""
    height, width = original.shape[:2]
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    detections = _eye_detector().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4,
        minSize=(max(40, int(min(height, width) * 0.16)),) * 2,
    )
    return any(w >= width * 0.20 and h >= height * 0.20 for _, _, w, h in detections)


def assert_eye_image(model, image, original):
    """Reject non-eye images while allowing normal close-up eye photographs."""
    if REFERENCE_PATH.exists():
        profile = np.load(REFERENCE_PATH)
        vectors = profile["vectors"]
        threshold = 0.38
    else:
        vectors, threshold = _make_eye_reference(model)
    candidate = _normalise(_embedding_model(model).predict(image[None, ...], verbose=0))
    similarity = float((candidate @ vectors.T).max())
    if similarity < threshold and not _contains_large_eye(original):
        raise ValueError("Only close-up eye photographs are accepted. This image does not match the eye-image screening dataset.")


def make_gradcam(model, image, class_index):
    import tensorflow as tf
    try:
        target_layer = model.get_layer("resnet50").get_layer("conv5_block3_out")
    except ValueError:
        target_layer = next(layer for layer in reversed(model.layers) if len(layer.output.shape) == 4)
    backbone = model.get_layer("resnet50")
    grad_model = tf.keras.models.Model(backbone.inputs, [target_layer.output, backbone.output])
    tensor = tf.convert_to_tensor(image[None, ...])
    with tf.GradientTape() as tape:
        x = model.get_layer("sequential")(tensor, training=False)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        conv_output, x = grad_model(x)
        for layer in model.layers[3:]:
            x = layer(x, training=False)
        loss = x[:, class_index]
    grads = tape.gradient(loss, conv_output)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_output[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return cv2.resize(heatmap.numpy(), (224, 224))


def predict(raw: bytes):
    model = load_model()
    if model is None:
        raise FileNotFoundError("No trained model found. Run training/train.py first.")
    original, image = preprocess(raw)
    assert_eye_image(model, image, original)
    probs = model.predict(image[None, ...], verbose=0)[0]
    index = int(np.argmax(probs))
    heatmap = make_gradcam(model, image, index)
    overlay = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    visual = cv2.addWeighted(cv2.resize(original, (224, 224)), 0.58, overlay, 0.42, 0)
    output = io.BytesIO()
    Image.fromarray(visual).save(output, format="PNG")
    return {"stage": CLASSES[index], "confidence": float(probs[index]), "probabilities": dict(zip(CLASSES, map(float, probs))), "heatmap": output.getvalue()}
