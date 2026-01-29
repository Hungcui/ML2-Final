import joblib
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

MODEL_PATH = r"D:\ML2-Final\saved_models\mnist_pca95_rf.joblib"
TEST_DIR   = r"D:\ML2-Final\test_images\39_05770.png"

model = joblib.load(MODEL_PATH)

def preprocess_to_mnist_vector(img_path: str) -> np.ndarray:
    """
    Returns shape (1, 784) float32, in 0..255 scale (matching your training input).
    """
    img = Image.open(img_path).convert("L")          # grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    arr = np.array(img, dtype=np.float32)            # PIL -> numpy :contentReference[oaicite:1]{index=1}

    # Optional auto-invert:
    # MNIST digits are bright on dark background; if your image is dark-on-white, invert it.
    # Heuristic: if background is mostly bright, invert.
    if arr.mean() > 127:
        arr = 255.0 - arr

    x = arr.reshape(1, -1)                           # (1, 784)
    return x

img_path = r"D:\ML2-Final\test_images\39_05770.png"
x = preprocess_to_mnist_vector(img_path)
print("Predicted digit:", model.predict(x)[0])

