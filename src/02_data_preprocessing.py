# 02_data_preprocessing.py
import json, pathlib, tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 1337

def get_datasets():
    root = pathlib.Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    train_root = raw_dir / "rps"
    test_root  = raw_dir / "rps-test-set"

    train_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_root, validation_split=VAL_SPLIT, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    class_names = train_ds_raw.class_names

    val_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_root, validation_split=VAL_SPLIT, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")
    test_ds_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_root, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
        label_mode="int")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds_raw.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_ds_raw.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds_raw.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

def write_labels_json(class_names):
    models_dir = pathlib.Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    with open(models_dir / "labels.json", "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)
    print("Saved:", models_dir / "labels.json")

def main():
    train_ds, val_ds, test_ds, class_names = get_datasets()
    print("Classes:", class_names)
    write_labels_json(class_names)
    n_train = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
    n_val   = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE
    n_test  = tf.data.experimental.cardinality(test_ds).numpy() * BATCH_SIZE
    print(f"Approx samples — train {n_train}, val {n_val}, test {n_test}")

if __name__ == "__main__":
    main()