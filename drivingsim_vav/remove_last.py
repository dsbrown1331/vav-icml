from pathlib import Path

import fire
import numpy as np


def main(datadir: str):
    data = Path(datadir)
    inputs = np.load(data / "inputs.npy")
    input_features = np.load(data / "input_features.npy")
    normals = np.load(data / "normals.npy")
    preferences = np.load(data / "preferences.npy")

    print(inputs.shape)

    assert inputs.shape[0] == input_features.shape[0] + 1
    assert input_features.shape[0] == normals.shape[0]
    assert normals.shape[0] == preferences.shape[0]

    inputs = inputs[:-2]
    input_features = input_features[:-1]
    normals = normals[:-1]
    preferences = preferences[:-1]

    print(inputs.shape)

    assert inputs.shape[0] == input_features.shape[0]
    assert input_features.shape[0] == normals.shape[0]
    assert normals.shape[0] == preferences.shape[0]

    np.save(data / "inputs.npy", inputs)
    np.save(data / "input_features.npy", input_features)
    np.save(data / "normals.npy", normals)
    np.save(data / "preferences.npy", preferences)


if __name__ == "__main__":
    fire.Fire(main)
