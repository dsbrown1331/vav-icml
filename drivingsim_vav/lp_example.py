from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from post import filter_halfplanes, sample
from run_tests import normalize, run_test


def main():
    reward = np.load(Path("questions/1/true_reward.npy"))
    normals = np.load(Path("questions/1/normals.npy"))
    preferences = np.load(Path("questions/1/preferences.npy"))

    normals = (normals.T * preferences).T

    rewards = sample(
        reward_dimension=4, normals=(normals.T * preferences).T, n_samples=100,
    )

    reward_noise = 0.01
    n_rewards = 100
    dist = multivariate_normal(mean=reward, cov=np.eye(reward.shape[0]) * reward_noise)

    fake_rewards = dist.rvs(n_rewards)
    fake_rewards = normalize(fake_rewards)

    print("Doing filtering with LP")
    filtered_normals, _ = filter_halfplanes(
        normals,
        preferences,
        skip_redundancy_filtering=False,
        rewards=rewards,
        deterministic=True,
    )
    print(f"There are {len(filtered_normals)} halfplanes after filtering.")
    frac_pass = np.mean(run_test(normals=filtered_normals, fake_rewards=fake_rewards))
    print(
        f"With {len(filtered_normals)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )

    print("Doing filtering without LP")
    filtered_normals, _ = filter_halfplanes(
        normals,
        preferences,
        skip_redundancy_filtering=True,
        rewards=rewards,
        deterministic=True,
    )
    print(f"There are {len(filtered_normals)} halfplanes after filtering.")
    frac_pass = np.mean(run_test(normals=filtered_normals, fake_rewards=fake_rewards))
    print(
        f"With {len(filtered_normals)} questions, "
        f"{frac_pass * 100}% of the fake rewards passed the test."
    )


if __name__ == "__main__":
    main()
