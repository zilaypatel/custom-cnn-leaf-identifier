"""Leaf Classification AI Model - Inference Script.

Author: Asha M Tarsadia Institute of Computer Science & Technology Research
Implementation
Model: Custom Convolutional Neural Network (Sequential)
Classes: Lemon, Mango, Rosa-Senensis, Tulsi, Money Plant
"""

import argparse
import os
import sys
import tempfile

# Set non-intrusive environment variables before importing heavy ML libraries
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["KERAS_HOME"] = os.path.join(tempfile.gettempdir(), "keras")

import numpy as np
from PIL import Image

# Metadata and botanical details
CLASS_NAMES = ["Lemon", "Mango", "Rosa-Senensis", "Tulsi", "money"]

CLASS_DISPLAY = {
    "Lemon": "Lemon (Citrus limon)",
    "Mango": "Mango (Mangifera indica)",
    "Rosa-Senensis": "Hibiscus / China Rose (Hibiscus rosa-sinensis)",
    "Tulsi": "Holy Basil / Tulsi (Ocimum tenuiflorum)",
    "money": "Money Plant (Epipremnum aureum)",
}

BOTANICAL_INFO = {
    "Lemon": {
        "scientific_name": "Citrus limon",
        "family": "Rutaceae",
        "medicinal_uses": (
            "Rich in Vitamin C and citric acid. Promotes digestion, boosts"
            " immunity, and possesses antimicrobial properties."
        ),
    },
    "Mango": {
        "scientific_name": "Mangifera indica",
        "family": "Anacardiaceae",
        "medicinal_uses": (
            "Contains mangiferin, polyphenols, and flavonoids. Traditionally"
            " used to manage blood glucose and respiratory conditions."
        ),
    },
    "Rosa-Senensis": {
        "scientific_name": "Hibiscus rosa-sinensis",
        "family": "Malvaceae",
        "medicinal_uses": (
            "Used in Ayurveda for promoting hair health, cooling body heat, and"
            " managing cardiovascular health and blood pressure."
        ),
    },
    "Tulsi": {
        "scientific_name": "Ocimum tenuiflorum",
        "family": "Lamiaceae",
        "medicinal_uses": (
            "Considered 'Queen of Herbs' in Ayurveda. Powerful adaptogen with"
            " antiviral, immunomodulatory, and stress-relieving properties."
        ),
    },
    "money": {
        "scientific_name": "Epipremnum aureum",
        "family": "Araceae",
        "medicinal_uses": (
            "Highly effective air purifier recognized in NASA clean air"
            " studies; filters volatile organic compounds (VOCs) and indoor"
            " toxins."
        ),
    },
}

IMAGE_SIZE = (128, 128)


def load_model(model_path="leaf_model.h5"):
    """Loads the Keras leaf classification model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Import tensorflow/keras inside function for fast CLI startup
    try:
        import tensorflow as tf

        # Silence verbose TF warnings
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model via TensorFlow: {e}")
        raise


def preprocess_image(image_path):
    """Loads and preprocesses an image for inference (128x128, normalized to [0, 1])."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)
    return img_tensor


def predict_single(model, image_path):
    """Runs inference on a single image and returns formatted results."""
    tensor = preprocess_image(image_path)
    predictions = model.predict(tensor, verbose=0)[0]

    top_idx = int(np.argmax(predictions))
    top_class = CLASS_NAMES[top_idx]
    confidence = float(predictions[top_idx]) * 100

    results = {
        "image_path": image_path,
        "predicted_class": top_class,
        "display_name": CLASS_DISPLAY[top_class],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(predictions[i]) * 100
            for i in range(len(CLASS_NAMES))
        },
        "botanical_info": BOTANICAL_INFO[top_class],
    }
    return results


def print_prediction_report(res):
    """Prints a structured prediction report to terminal."""
    print("=" * 70)
    print("           INDIAN HERBAL LEAF CLASSIFICATION REPORT")
    print("=" * 70)
    print(f"Image File      : {res['image_path']}")
    print(f"Predicted Leaf  : {res['display_name']}")
    print(f"Confidence Score: {res['confidence']:.2f}%")
    print("-" * 70)
    print("Probability Distribution:")
    for cls_name in CLASS_NAMES:
        prob = res["probabilities"][cls_name]
        bar = "█" * int(prob / 4)
        print(
            f"  • {CLASS_DISPLAY[cls_name]:<42} : {prob:6.2f}% |"
            f" {bar:<25}"
        )
    print("-" * 70)
    info = res["botanical_info"]
    print(f"Scientific Name : {info['scientific_name']}")
    print(f"Botanical Family: {info['family']}")
    print(f"Medicinal Uses  : {info['medicinal_uses']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Predict leaf species using custom CNN AI model."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an individual leaf image (.jpg, .jpeg, .png)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Directory of images to classify in batch mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="leaf_model.h5",
        help="Path to trained .h5 model file (default: leaf_model.h5)",
    )

    args = parser.parse_args()

    # If no arguments provided, test with first available image in workspace
    if not args.image and not args.dir:
        for folder in CLASS_NAMES:
            if os.path.isdir(folder):
                files = [
                    f
                    for f in os.listdir(folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if files:
                    args.image = os.path.join(folder, files[0])
                    print(f"No image specified. Using sample image: {args.image}")
                    break

    if not args.image and not args.dir:
        print("Please provide an image with --image <path> or --dir <folder>.")
        sys.exit(1)

    print(f"Loading custom CNN model from '{args.model}'...")
    model = load_model(args.model)
    print("Model loaded successfully.\n")

    if args.image:
        res = predict_single(model, args.image)
        print_prediction_report(res)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Directory not found: {args.dir}")
            sys.exit(1)

        valid_exts = (".jpg", ".jpeg", ".png")
        image_files = [
            f for f in os.listdir(args.dir) if f.lower().endswith(valid_exts)
        ]
        if not image_files:
            print(f"No valid image files found in {args.dir}")
            sys.exit(0)

        print(
            f"Classifying {len(image_files)} images from '{args.dir}' in batch"
            " mode...\n"
        )
        print(f"{'File Name':<35} | {'Predicted Class':<25} | {'Confidence':<10}")
        print("-" * 75)
        for fname in image_files:
            fpath = os.path.join(args.dir, fname)
            res = predict_single(model, fpath)
            print(
                f"{fname:<35} | {res['predicted_class']:<25} |"
                f" {res['confidence']:.2f}%"
            )


if __name__ == "__main__":
    main()
