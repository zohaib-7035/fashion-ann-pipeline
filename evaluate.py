"""Stage 4: evaluate on the test set, write metrics.json + confusion matrix."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main():
    model = keras.models.load_model("models/model.h5")
    test = np.load("data/processed/test.npz")

    loss, acc = model.evaluate(test["x"], test["y"], verbose=0)
    preds = np.argmax(model.predict(test["x"], verbose=0), axis=1)

    cm = confusion_matrix(test["y"], preds)
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, xticks_rotation=45, colorbar=False)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/confusion_matrix.png", dpi=120)

    with open("metrics.json", "w") as f:
        json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, f, indent=2)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
