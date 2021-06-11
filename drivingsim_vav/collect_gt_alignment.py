from pathlib import Path
from threading import TIMEOUT_MAX
from typing import List, Optional

import fire  # type: ignore
import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.random import default_rng

from elicitation import append, load, make_mode_reward
from sampling import Sampler
from simulation_utils import compute_best, create_env


def make_path(reward: np.ndarray) -> np.ndarray:
    simulation_object = create_env("driver")
    optimal_ctrl = compute_best(simulation_object=simulation_object, w=reward, iter_count=10)
    return optimal_ctrl


def collect(
    outdir: Path,
    n_rewards: int,
    test_reward_path: Optional[Path] = None,
    std: Optional[float] = None,
    mean_reward_path: Optional[Path] = None,
    normals_paths: Optional[List[Path]] = None,
    preferences_paths: Optional[List[Path]] = None,
    use_random: bool = False,
    use_plausible: bool = False,
    skip_human: bool = False,
    overwrite: bool = False,
) -> None:
    """Collects ground truth labels for the optimal trajectories of some reward functions.

    Args:
        outdir (Path): Directory to write output to
        n_rewards (int): Number of rewards to generate or process
        test_reward_path (Optional[Path], optional): Path to nupmy array of reward weights to test. Defaults to None.
        std (Optional[float], optional): Standard deviation of normal distribution to draw test reward weigths from. Defaults to None.
        mean_reward_path (Optional[Path], optional): Path to numpy array specifying mean reward weights to sample around. Defaults to None.
        overwrite (bool, optional): Overwrite output? Defaults to False.

    Raises:
        ValueError: Raised if neither test_reward_path or both std and mean_reward_path are specified. The test rewards need to come from somewhere.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_rewards = load(outdir, "test_rewards.npy", overwrite=overwrite)
    new_rewards_index = out_rewards.shape[0] if out_rewards is not None else 0
    num_new_rewards = n_rewards - new_rewards_index

    if num_new_rewards > 0:
        if test_reward_path is not None:
            rewards = np.load(test_reward_path)[new_rewards_index:num_new_rewards]
        elif mean_reward_path is not None and std is not None:
            mean_reward = np.load(mean_reward_path)
            rewards = default_rng().normal(
                loc=mean_reward, scale=std, size=(num_new_rewards, *mean_reward.shape)
            )
        elif normals_paths is not None and preferences_paths is not None and std is not None:
            # NOTE(AUTHOR_1): This turned out not to work, because the random baseline is poisoning the well
            normals = None
            for normals_path, preferences_path in zip(normals_paths, preferences_paths):
                single_normals = np.load(normals_path)
                single_preferences = np.load(preferences_path)
                single_normals = (single_normals.T * single_preferences).T
                normals = append(normals, single_normals, flat=True)
            # TODO(AUTHOR_1): These can all be loaded in from flags.pkl, but I'm too lazy for that.
            mean_reward = make_mode_reward(
                query_type="strict",
                true_delta=1.1,
                w_sampler=Sampler(create_env("driver").num_of_features),
                n_reward_samples=100,
            )
            assert np.all(np.isfinite(mean_reward))
            rewards = default_rng().normal(
                loc=mean_reward, scale=std, size=(num_new_rewards, *mean_reward.shape)
            )
            assert np.all(np.isfinite(rewards))
        elif use_random:
            rewards = default_rng().normal(
                loc=0, scale=1, size=(num_new_rewards, create_env("driver").num_of_features)
            )
            rewards = rewards / np.linalg.norm(rewards)
        elif use_plausible:
            # Generate uniform rewards with plausible weights i.e. ones with the right sign
            rewards = default_rng().normal(
                loc=0, scale=1, size=(num_new_rewards, create_env("driver").num_of_features)
            )
            rewards = rewards / np.linalg.norm(rewards)

            # See models.py for reward feature details.
            rewards[:, 0] = np.abs(rewards[:, 0])
            rewards[:, 1] = -np.abs(rewards[:, 1])
            rewards[:, 2] = np.abs(rewards[:, 2])
            rewards[:, 3] = -np.abs(rewards[:, 3])
        else:
            raise ValueError(
                "You must either supply a path to the test rewards, or a mean reward and "
                "std from which to sample the test rewards."
            )
        out_rewards = append(out_rewards, rewards, flat=True)
    else:
        assert out_rewards is not None

    assert np.all(np.isfinite(out_rewards))
    np.save(open(outdir / "test_rewards.npy", "wb"), out_rewards)

    paths = load(outdir, "optimal_paths.npy", overwrite=overwrite)
    new_paths_index = paths.shape[0] if paths is not None else 0
    num_new_paths = n_rewards - new_paths_index

    if num_new_paths > 0:
        new_paths = np.array(
            Parallel(n_jobs=-2)(
                delayed(make_path)(reward) for reward in out_rewards[new_paths_index:]
            )
        )
        paths = append(paths, new_paths, flat=True)
    else:
        assert paths is not None
    np.save(open(outdir / "optimal_paths.npy", "wb"), np.array(paths))

    gt_alignment = load(outdir, "alignment.npy", overwrite=overwrite)
    new_gt_index = gt_alignment.size if gt_alignment is not None else 0

    if skip_human:
        exit()

    simulation_object = create_env("driver")
    for path in paths[new_gt_index:]:
        simulation_object.set_ctrl(path)
        simulation_object.watch(1)

        alignment = input("Aligned (y/n):").lower()
        while alignment not in ["y", "n"]:
            alignment = input("Aligned (y/n):").lower()
        gt_alignment = append(gt_alignment, alignment == "y")

    np.save(open(outdir / "alignment.npy", "wb"), gt_alignment)


if __name__ == "__main__":
    fire.Fire(collect)
