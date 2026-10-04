"""Stage 2: normalize pixels to [0, 1] and split a validation set."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load("data/raw/train.npz")
    test = np.load("data/raw/test.npz")

    # Normalization step (min-max scaling to [0, 1])
    x_train = train["x"].astype("float32") / 255.0
    x_test = test["x"].astype("float32") / 255.0

    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, train["y"],
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=train["y"],
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_tr, y=y_tr)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=test["y"])
    print(f"train {x_tr.shape} | val {x_val.shape} | test {x_test.shape}")


if __name__ == "__main__":
    main()
