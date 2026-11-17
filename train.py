"""Leaf Classification AI Model - Training Pipeline.

Author: Asha M Tarsadia Institute of Computer Science & Technology Research
Implementation
Model: Custom Convolutional Neural Network (Sequential)
Paper: 'Leaf Classification Using A Custom CNN'
Features:
- Exact 3-block Conv2D + MaxPool + Dense(256) + Dropout(0.5) architecture (13.8M
params)
- Automatic Data Augmentation (Rotation, Horizontal Flip, Brightness)
- Class Weighting / Rebalancing to handle minority classes (Tulsi, Mango, Money
Plant)
- Training curves plotting (accuracy & loss) saved as training_history.png
"""

import argparse
import os
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["KERAS_HOME"] = os.path.join(tempfile.gettempdir(), "keras")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

CLASS_NAMES = ["Lemon", "Mango", "Rosa-Senensis", "Tulsi", "money"]
IMAGE_SIZE = (128, 128)


def build_custom_cnn(input_shape=(128, 128, 3), num_classes=5):
    """Builds the exact Custom CNN architecture from the research paper.

    Architecture:
    - Layer 1: Conv2D(32, 3x3) -> ReLU -> MaxPooling2D(2x2)
    - Layer 2: Conv2D(64, 3x3) -> ReLU -> MaxPooling2D(2x2)
    - Layer 3: Conv2D(128, 3x3) -> ReLU -> MaxPooling2D(2x2)
    - Flatten Layer
    - Dense(256) -> ReLU -> Dropout(0.5)
    - Output: Dense(5) -> Softmax
    Total Parameters: 13,872,707
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="valid"),
            layers.MaxPooling2D((2, 2)),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="valid"),
            layers.MaxPooling2D((2, 2)),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="valid"),
            layers.MaxPooling2D((2, 2)),
            # Dense Classification Head
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="Leaf_Custom_CNN",
    )
    return model


def get_data_augmentation():
    """Returns safe data augmentation layers followed by normalization to [0, 1]."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.Rescaling(1.0 / 255.0),
        ],
        name="data_augmentation",
    )


def compute_dataset_class_weights(dataset_dir="."):
    """Computes safely normalized class weights across the 5 plant leaf classes.

    Normalizes weights so the average weight is 1.0, preventing extreme gradient
    instability for very small sample classes while still countering imbalance.
    """
    class_counts = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(dataset_dir, cls)
        if os.path.isdir(cls_dir):
            count = len(
                [
                    f
                    for f in os.listdir(cls_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
            class_counts.append(max(count, 1))
        else:
            class_counts.append(1)

    total_samples = sum(class_counts)
    num_classes = len(CLASS_NAMES)
    raw_weights = [
        total_samples / (num_classes * count) for count in class_counts
    ]
    # Smooth with square root to prevent destabilizing gradient steps
    smoothed_weights = [np.sqrt(w) for w in raw_weights]
    norm_factor = num_classes / sum(smoothed_weights)
    class_weights = {
        i: float(smoothed_weights[i] * norm_factor) for i in range(num_classes)
    }

    print("=" * 60)
    print("Dataset Class Distribution & Smoothed Balanced Weights:")
    for idx, (cls, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        print(
            f"  • Class {idx} ({cls:<14}): {count:4d} images | Weight:"
            f" {class_weights[idx]:.2f}"
        )
    print("=" * 60)
    return class_weights


def load_datasets(dataset_dir=".", val_split=0.2, batch_size=32, seed=42):
    """Loads and splits the image dataset from directory structure."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )

    # Augmentation with normalization included
    data_augmentation = get_data_augmentation()
    val_rescale = layers.Rescaling(1.0 / 255.0)

    # Apply preprocessing to training pipeline
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (val_rescale(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds


def plot_training_history(history, save_path="training_history.png"):
    """Plots and exports loss and accuracy curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2563EB", lw=2)
    plt.plot(
        epochs_range,
        val_acc,
        label="Validation Accuracy",
        color="#16A34A",
        lw=2,
        ls="--",
    )
    plt.title("Model Accuracy over Epochs", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1.05])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#DC2626", lw=2)
    plt.plot(
        epochs_range,
        val_loss,
        label="Validation Loss",
        color="#D97706",
        lw=2,
        ls="--",
    )
    plt.title("Categorical Cross-Entropy Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Custom CNN Model for Indian Herbal Leaf Classification."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Dataset directory containing class subdirectories",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="leaf_model_trained.h5",
        help="Path to save trained .h5 model file (default: leaf_model_trained.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick 1-epoch dry run to test pipeline execution",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("[DRY-RUN MODE] Running 1 epoch with reduced steps.")
        args.epochs = 1

    print("\n--- Initializing Data Pipeline ---")
    class_weights = compute_dataset_class_weights(args.data_dir)
    train_ds, val_ds = load_datasets(
        dataset_dir=args.data_dir,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )

    print("\n--- Constructing Custom CNN Architecture ---")
    model = build_custom_cnn(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(CLASS_NAMES),
    )
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.output_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(
        f"\n--- Starting Model Training ({args.epochs} epochs, batch size"
        f" {args.batch_size}) ---"
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # Plot training metrics
    plot_training_history(history, save_path="training_history.png")

    print(f"\nTraining complete. Best model checkpoint saved to: {args.output_model}")


if __name__ == "__main__":
    main()
