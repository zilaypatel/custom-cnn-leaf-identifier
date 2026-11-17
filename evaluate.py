"""Leaf Classification AI Model - Evaluation Pipeline.

Author: Asha M Tarsadia Institute of Computer Science & Technology Research
Implementation
Model: Custom Convolutional Neural Network (Sequential)
Outputs:
- Test Set Accuracy & Cross-Entropy Loss
- Per-class Precision, Recall, and F1-Score
- Confusion Matrix Heatmap saved as confusion_matrix.png
"""

import argparse
import os
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["KERAS_HOME"] = os.path.join(tempfile.gettempdir(), "keras")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

CLASS_NAMES = ["Lemon", "Mango", "Rosa-Senensis", "Tulsi", "money"]

CLASS_DISPLAY = [
    "Lemon",
    "Mango",
    "Rosa-Senensis",
    "Tulsi",
    "Money Plant",
]

IMAGE_SIZE = (128, 128)


def load_model(model_path="leaf_model.h5"):
    """Loads the trained Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return tf.keras.models.load_model(model_path)


def gather_test_samples(data_dir=".", max_per_class=None, val_ratio=0.2):
    """Collects sample test images from each class directory."""
    images = []
    labels = []
    file_paths = []

    print("Collecting test images from dataset folders...")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            continue

        files = sorted(
            [
                f
                for f in os.listdir(class_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        # Take validation slice (last val_ratio proportion of files)
        start_idx = int(len(files) * (1.0 - val_ratio))
        val_files = files[start_idx:]

        if max_per_class and len(val_files) > max_per_class:
            val_files = val_files[:max_per_class]

        print(
            f"  • {class_name:<15}: {len(val_files)} evaluation samples"
            f" (from {len(files)} total)"
        )

        for fname in val_files:
            fpath = os.path.join(class_folder, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_idx)
                file_paths.append(fpath)
            except Exception as e:
                print(f"Skipping corrupt image {fpath}: {e}")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y, file_paths


def plot_confusion_matrix(
    cm,
    class_names=CLASS_DISPLAY,
    save_path="confusion_matrix.png",
):
    """Generates and saves a clean confusion matrix heatmap plot."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize CM by true label rows for percentages
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Custom CNN Leaf Classification - Confusion Matrix",
        ylabel="True Botanical Class",
        xlabel="Predicted Botanical Class",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
    )
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    # Annotate counts and percentages inside each cell
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Custom CNN model on dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="leaf_model.h5",
        help="Path to trained .h5 model file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Dataset root directory containing class folders",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Max evaluation samples per class (for quick evaluation)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation/Test slice proportion from folders (default: 0.2)",
    )

    args = parser.parse_args()

    print(f"Loading model from '{args.model}'...")
    model = load_model(args.model)

    X_test, y_true, file_paths = gather_test_samples(
        data_dir=args.data_dir,
        max_per_class=args.sample_size,
        val_ratio=args.val_ratio,
    )

    if len(X_test) == 0:
        print("No test samples found to evaluate.")
        return

    print(f"\nRunning model inference on {len(X_test)} evaluation images...")
    predictions = model.predict(X_test, batch_size=32, verbose=1)
    y_pred = np.argmax(predictions, axis=1)

    # Compute Overall Accuracy
    overall_acc = np.mean(y_pred == y_true) * 100
    print("\n" + "=" * 65)
    print(f"  EVALUATION RESULTS: OVERALL TEST ACCURACY: {overall_acc:.2f}%")
    print("=" * 65)

    # Classification Report
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_DISPLAY,
            labels=list(range(len(CLASS_NAMES))),
            digits=4,
            zero_division=0,
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    plot_confusion_matrix(cm, save_path="confusion_matrix.png")


if __name__ == "__main__":
    main()
