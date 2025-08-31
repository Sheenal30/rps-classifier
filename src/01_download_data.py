# 01_download_data.py
# Downloads the Rock, Paper, Scissors dataset and unzips into data/raw/
# Source: TensorFlow tutorial dataset by Laurence Moroney
import pathlib
import tensorflow as tf

DATASETS = {
    # new mirror under download.tensorflow.org/data
    "rps": "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
    "rps-test-set": "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip",
}

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DATASETS.items():
        print(f"Downloading {name} ...")
        zip_path = tf.keras.utils.get_file(
            fname=f"{name}.zip",
            origin=url,
            cache_dir=str(raw_dir),
            extract=True,
            archive_format="zip",
        )
        # tf.keras.utils.get_file extracts to cache_dir/datasets/<name>/...
        extracted = pathlib.Path(zip_path).with_suffix("")
        # Move extracted folder to data/raw/<name>
        # Depending on TF version, extract path may be either .../datasets/<name> or .../<name>
        candidates = [
            raw_dir / "datasets" / name,
            raw_dir / name,
        ]
        target = raw_dir / name
        for c in candidates:
            if c.exists() and c.is_dir():
                if target.exists():
                    pass
                else:
                    c.rename(target)
        print(f"Ready: {target}")

    print("\nFolder structure under data/raw:")
    for p in sorted((raw_dir).rglob("*")):
        if p.is_dir() and p.parent.name in {"rps","rps-test-set"}:
            print(p)

if __name__ == "__main__":
    main()
