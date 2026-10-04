"""Stage 2: normalize pixels and split a validation set."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["preprocess"]

    train = np.load("data/raw/train.npz")
    test = np.load("data/raw/test.npz")

    # Normalization step (reconciled: method chosen in params.yaml)
    x_train = train["x"].astype("float32")
    x_test = test["x"].astype("float32")
    if params["normalization"] == "standard":      # teammate's approach
        mean, std = x_train.mean(), x_train.std()
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    elif params["normalization"] == "symmetric":   # main's approach
        x_train, x_test = x_train / 127.5 - 1.0, x_test / 127.5 - 1.0
    else:                                          # "minmax" (original)
        x_train, x_test = x_train / 255.0, x_test / 255.0

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
