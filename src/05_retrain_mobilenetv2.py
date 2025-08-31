# src/05_retrain_mobilenetv2.py
# Two-stage MobileNetV2 training, batch-safe augmentations, no Lambda preprocess.
import pathlib, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
VAL_SPLIT  = 0.2
SEED       = 1337

def load_datasets():
    root       = pathlib.Path(__file__).resolve().parents[1]
    train_dir  = root / "data" / "raw" / "rps"
    extra_dir  = root / "data" / "real"
    test_dir   = root / "data" / "raw" / "rps-test-set"

    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=VAL_SPLIT, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")

    class_names = list(train_raw.class_names)

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, validation_split=VAL_SPLIT, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="int")

    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False, label_mode="int")

    train_combined = train_raw
    if extra_dir.exists():
        try:
            extra = tf.keras.preprocessing.image_dataset_from_directory(
                extra_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
                shuffle=True, label_mode="int", class_names=class_names)
            train_combined = train_raw.concatenate(extra)
            print("Found extra real data. Concatenated to training set.")
        except Exception as e:
            print("Warning: failed to load extra_dir with explicit class_names. Skipping extra_dir.", e)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_combined.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    val_ds   = val_raw.cache().prefetch(AUTOTUNE)
    test_ds  = test_raw.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, test_ds, class_names

def build_model(n_classes):
    # batch-safe augmentation
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomCrop(int(IMG_SIZE[0]*0.9), int(IMG_SIZE[1]*0.9)),
        tf.keras.layers.Resizing(IMG_SIZE[0], IMG_SIZE[1]),
        tf.keras.layers.RandomContrast(0.20),
    ], name="data_augmentation")

    # Replace Lambda(preprocess) with Rescaling to avoid Python-lambda layer
    # preprocess_input for MobileNetV2 = x / 127.5 - 1
    preprocessing_layer = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale_to_mnv2")

    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape=IMG_SIZE + (3,))
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_aug(inputs)
    x = preprocessing_layer(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base

def main():
    root       = pathlib.Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets()

    # Slightly upweight scissors if present
    class_weight = {i: 1.0 for i in range(len(class_names))}
    if "scissors" in class_names:
        class_weight[class_names.index("scissors")] = 1.5

    json.dump({"class_names": class_names}, open(models_dir / "labels.json", "w"), indent=2)

    model, base = build_model(len(class_names))

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(models_dir / "best_rps_mobilenetv2.h5"),
        save_best_only=True, monitor="val_accuracy", mode="max")

    print("Stage 1: training with frozen base")
    model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=[ckpt],
              class_weight=class_weight)

    print("Stage 2: fine-tune top layers")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[ckpt],
              class_weight=class_weight)

    # Save final models
    model.save(models_dir / "best_rps_mobilenetv2.h5")
    model.save(models_dir / "best_rps_mobilenetv2.keras")

    # Evaluate using the new saved .keras (no Lambda layers now)
    best = tf.keras.models.load_model(models_dir / "best_rps_mobilenetv2.keras", compile=False)

    y_true, y_pred = [], []
    for x, y in test_ds:
        y_true.extend(y.numpy().tolist())
        preds = best.predict(x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("Classification report\n", rep)
    with open(models_dir / "metrics_report.txt", "w") as f:
        f.write(rep)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names, ylabel='True', xlabel='Predicted')
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(models_dir / "confusion_matrix.png", dpi=160)
    print("Saved:", models_dir / "confusion_matrix.png")
    print("Saved:", models_dir / "best_rps_mobilenetv2.keras")
    print("Saved:", models_dir / "metrics_report.txt")

if __name__ == "__main__":
    main()