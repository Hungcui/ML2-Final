import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from tensorflow import keras  

MODEL_PATH = r"D:\ML2-Final\saved_models\mnist_cnn.keras"  
TEST_DIR   = Path(r"D:\ML2-Final\test_images")

# Load model if not already defined
try:
    model
except NameError:
    model = keras.saving.load_model(MODEL_PATH)  # loads model saved via model.save()

def preprocess_for_cnn(img_path: str) -> np.ndarray:
    """
    Output: (1, 28, 28, 1) float32 in [0, 1]
    Assumes MNIST-style: bright digit on dark background.
    """
    img = Image.open(img_path)

    # Convert to grayscale (handles RGB/RGBA/etc.)
    img = img.convert("L")

    # Resize with a good resampling filter
    img = img.resize((28, 28), resample=Image.Resampling.LANCZOS)  

    arr = np.array(img, dtype=np.float32) / 255.0

    # Auto-invert if background is white-ish (common for drawn digits)
    # If average pixel is bright, invert so digit becomes bright on dark.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr.reshape(1, 28, 28, 1)

# Collect images (supports adding more later; rerun to include new files)
exts = {".png", ".jpg", ".jpeg", ".bmp"}
image_paths = sorted([p for p in TEST_DIR.iterdir() if p.suffix.lower() in exts])

if not image_paths:
    print(f"No images found in: {TEST_DIR}")
else:
    for p in image_paths:
        try:
            x = preprocess_for_cnn(str(p))
            probs = model.predict(x, verbose=0)          
            pred = int(np.argmax(probs, axis=1)[0])
            conf = float(np.max(probs))
            print(f"{p.name} -> predicted: {pred} (confidence={conf:.4f})")
        except Exception as e:
            print(f"{p.name} -> ERROR: {e}")
