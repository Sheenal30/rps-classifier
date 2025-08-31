# predict.py
import sys, pathlib, numpy as np, tensorflow as tf
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_SIZE = (224, 224)

def load_labels():
    import json, pathlib
    return json.load(open(pathlib.Path("models") / "labels.json"))["class_names"]

def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE)  # keep aspect
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    return arr

def main(folder):
    model = tf.keras.models.load_model("models/best_rps_mobilenetv2.keras",
                                       compile=False)
    class_names = load_labels()
    for p in sorted(pathlib.Path(folder).glob("*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        logits = model.predict(preprocess(p), verbose=0)[0]
        idx = int(np.argmax(logits))
        print(f"{p.name:30s} → {class_names[idx]:8s}  {logits[idx]:.2%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <folder-of-images>")
        sys.exit(1)
    main(sys.argv[1])