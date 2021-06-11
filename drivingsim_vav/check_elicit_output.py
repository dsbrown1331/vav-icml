# A series of sanity tests for output from the elicitation process.
import logging
import pickle
from pathlib import Path

import fire  # type: ignore
import numpy as np

from simulation_utils import create_env
from utils import assert_normals, assert_reward, get_mean_reward, orient_normals


def make_normals(input_features: np.ndarray) -> np.ndarray:
    normals = input_features[:, 0] - input_features[:, 1]
    assert_normals(normals, False, input_features.shape[2])
    return normals


def make_input_features(inputs: np.ndarray, sim) -> np.ndarray:
    input_features = np.empty((inputs.shape[0], 2, sim.num_of_features))
    for i, (a, b) in enumerate(inputs):
        sim.feed(a)
        input_features[i, 0] = sim.get_features()

        sim.feed(b)
        input_features[i, 1] = sim.get_features()

    return input_features


def assert_input_feature_consistency(inputs: np.ndarray, input_features: np.ndarray, sim) -> None:
    recreated_input_features = make_input_features(inputs, sim)
    matches = recreated_input_features == input_features
    if not np.all(matches):
        bad_indices = np.logical_not(matches)
        bad_inputs = inputs[bad_indices]
        bad_input_features = input_features[bad_indices]
        expected_bad_outputs = recreated_input_features[bad_indices]
        logging.error("Some input features don't match the recreated inputs.")
        logging.error(f"The following inputs are bad:\n{bad_inputs}")
        logging.error(f"The recorded features for these inputs are:\n{bad_input_features}")
        logging.error(f"The recreated input_features are:\n{expected_bad_outputs}")
        logging.error(f"The bad indices are {np.where(bad_indices)}")
        assert np.all(matches)


def assert_normal_consistency(input_features: np.ndarray, normals: np.ndarray) -> None:
    assert np.all(make_normals(input_features) == normals)


def assert_true_reward_consistency(oriented_normals: np.ndarray, true_reward: np.ndarray) -> None:
    gt_value_diff = oriented_normals @ true_reward
    pref_correct = gt_value_diff >= 0
    if not np.all(pref_correct):
        pref_incorrect = np.logical_not(pref_correct)
        bad_normals = oriented_normals[pref_incorrect]
        bad_values = gt_value_diff[pref_incorrect]
        logging.error("Some preferences backwards relative to gt reward.")
        logging.error(f"The following normals are bad:\n{bad_normals}")
        logging.error(f"The value difference for these normals were:\n{bad_values}")
        logging.error(f"The ground truth reward is {true_reward}")
        logging.error(f"The bad normal indices are {np.where(pref_incorrect)}")
        assert np.all(pref_correct)


def main(datadir: Path) -> None:
    logging.basicConfig(level="INFO")

    datadir = Path(datadir)

    flags = pickle.load(open(datadir / "flags.pkl", "rb"))
    use_equiv = False
    sim = create_env(flags["task"])
    n_reward_features = sim.num_of_features

    inputs = np.load(datadir / "inputs.npy")
    n_questions = inputs.shape[0]
    assert inputs.shape[1] == 2

    input_features = np.load(datadir / "input_features.npy")
    n_questions = input_features.shape[0]
    assert input_features.shape == (n_questions, 2, n_reward_features), input_features.shape

    assert_input_feature_consistency(inputs, input_features, sim)

    normals = np.load(datadir / "normals.npy")
    logging.info(f"There are {normals.shape[0]} questions")
    assert_normals(normals, use_equiv, n_reward_features)

    assert_normal_consistency(input_features, normals)

    preferences = np.load(datadir / "preferences.npy")
    assert preferences.shape == (n_questions,)
    assert np.all((preferences == 1) | (preferences == -1))

    oriented_normals = orient_normals(normals, preferences)

    if (datadir / "true_reward.npy").exists():
        true_reward = np.load(datadir / "true_reward.npy")
        assert_reward(true_reward, use_equiv, n_reward_features)
        logging.info(true_reward)
        assert_true_reward_consistency(oriented_normals, true_reward)

    if (datadir / "mean_reward.npy").exists():
        mean_reward = np.load(datadir / "mean_reward.npy")
        logging.info(mean_reward)
        assert_reward(mean_reward, use_equiv, n_reward_features)

        mean_accuracy = np.mean(oriented_normals @ mean_reward > 0)
        logging.info(f"Accuracy of mean reward function is {mean_accuracy}")


if __name__ == "__main__":
    fire.Fire(main)
