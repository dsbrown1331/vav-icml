# Computes statistics of the ground truth reward gap between trajectories
# in order to compare with epsilon.

from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from utils import orient_normals


def main(
    datadir=Path("volume-data/questions"),
    n_replications=10,
    reward_name=Path("true_reward.npy"),
    questions_name=Path("normals.npy"),
    preferences_name=Path("preferences.npy"),
):
    gaps = None
    for replication in range(1, 1 + n_replications):
        reward = np.load(datadir / str(replication) / reward_name)
        questions = np.load(datadir / str(replication) / questions_name)
        preferences = np.load(datadir / str(replication) / preferences_name)
        normals = orient_normals(questions, preferences)
        true_value_gap = normals @ reward
        gaps = np.concatenate((gaps, true_value_gap)) if gaps is not None else true_value_gap

    print(f"Mean reward gap: {np.mean(gaps)} ({np.std(gaps)})")
    print(f"20/80th percentiles: {np.percentile(gaps, 5)}, {np.percentile(gaps, 95)}")
    # plt.hist(gaps, cumulative=True, density=True, bins=1000)
    # plt.xlabel("Value gap")
    # plt.title("Histogram of ground truth value gaps")
    # plt.show()
    # plt.close()


if __name__ == "__main__":
    main()
