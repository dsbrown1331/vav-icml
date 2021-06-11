import logging
import pickle as pkl
from pathlib import Path

import fire
import numpy as np
from numpy.linalg import norm

from sampling import Sampler


def assert_nonempty(*arrs) -> None:
    for arr in arrs:
        assert len(arr) > 0


def assert_normals(normals: np.ndarray, use_equiv: bool, n_reward_features: int = 4) -> None:
    """ Asserts the given array is an array of normal vectors defining half space constraints."""
    shape = normals.shape
    assert len(shape) == 2, f"shape does not have 2 dimensions:{shape}"
    # Constant offset constraint adds one dimension to normal vectors.
    assert shape[1] == n_reward_features + int(use_equiv)


def assert_reward(
    reward: np.ndarray, use_equiv: bool, n_reward_features: int = 4, eps: float = 0.000_001
) -> None:
    """ Asserts the given array is might be a reward feature vector. """
    assert np.all(np.isfinite(reward))
    assert reward.shape == (n_reward_features + int(use_equiv),)
    assert abs(norm(reward) - 1) < eps


def assert_rewards(
    rewards: np.ndarray, use_equiv: bool, n_reward_features: int = 4, eps: float = 0.000_001
) -> None:
    assert np.all(np.isfinite(rewards))
    assert len(rewards.shape) == 2
    assert rewards.shape[1] == n_reward_features + int(
        use_equiv
    ), f"rewards.shape={rewards.shape}, n_reward_features={n_reward_features}, use_equiv={use_equiv}"
    norm_dist = abs(norm(rewards, axis=1) - 1)
    norm_errors = norm_dist > eps
    if np.any(norm_errors):
        logging.error("Some rewards are not normalized")
        indices = np.where(norm_errors)
        logging.error(f"Bad distances:\n{norm_dist[indices]}")
        logging.error(f"Bad rewards:\n{rewards[indices]}")
        logging.error(f"Bad indices:\n{indices}")
        assert not np.any(norm_errors)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """ Takes in a 2d array of row vectors and ensures each row vector has an L_2 norm of 1."""
    return (vectors.T / norm(vectors, axis=1)).T


def orient_normals(
    normals: np.ndarray,
    preferences: np.ndarray,
    use_equiv: bool = False,
    n_reward_features: int = 4,
) -> np.ndarray:
    assert_normals(normals, use_equiv, n_reward_features)
    assert preferences.shape == (normals.shape[0],)

    oriented_normals = (normals.T * preferences).T

    assert_normals(oriented_normals, use_equiv, n_reward_features)
    return oriented_normals


def get_mean_reward(
    elicited_input_features: np.ndarray,
    elicited_preferences: np.ndarray,
    M: int,
    query_type: str,
    delta: float,
):
    n_features = elicited_input_features.shape[2]
    w_sampler = Sampler(n_features)
    for (a_phi, b_phi), preference in zip(elicited_input_features, elicited_preferences):
        w_sampler.feed(a_phi, b_phi, [preference])
    reward_samples, _ = w_sampler.sample_given_delta(M, query_type, delta)
    mean_reward = np.mean(reward_samples, axis=0)
    assert len(mean_reward.shape) == 1 and mean_reward.shape[0] == n_features
    return mean_reward


# Jank functions for directly modifying file output because I wrote a bug and don't want to re-run
# everything.


def flip_prefs(preferences_path: Path) -> None:
    preferences = np.load(preferences_path)
    preferences *= -1
    np.save(preferences_path, preferences)


def trim(n_questions: int, datadir: Path) -> None:
    datadir = Path(datadir)
    normals = np.load(datadir / "normals.npy")
    input_features = np.load(datadir / "input_features.npy")
    preferences = np.load(datadir / "preferences.npy")
    inputs = np.load(datadir / "inputs.npy")

    assert normals.shape[0] == input_features.shape[0]
    assert normals.shape[0] == preferences.shape[0]
    assert normals.shape[0] == inputs.shape[0]

    normals = normals[n_questions:]
    input_features = input_features[n_questions:]
    preferences = preferences[n_questions:]
    inputs = inputs[n_questions:]

    np.save(datadir / "normals.npy", normals)
    np.save(datadir / "input_features.npy", input_features)
    np.save(datadir / "preferences.npy", preferences)
    np.save(datadir / "inputs.npy", inputs)


def fix_flags(flags_path: Path):
    flags = pkl.load(open(flags_path, "rb"))
    if "equiv_size" not in flags.keys():
        flags["equiv_size"] = flags["delta"]
    pkl.dump(flags, open(flags_path, "wb"))


if __name__ == "__main__":
    fire.Fire()
