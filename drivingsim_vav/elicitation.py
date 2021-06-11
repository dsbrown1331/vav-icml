import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import fire  # type: ignore
import numpy as np
from joblib.parallel import Parallel, delayed

from sampling import Sampler
from simulation_utils import create_env, get_feedback, get_simulated_feedback, run_algo


def load(outdir: Path, filename: str, overwrite: bool) -> Optional[np.ndarray]:
    if overwrite:
        return None

    filepath = outdir / filename
    if filepath.exists():
        return np.load(filepath)

    return None


def make_mode_reward(
    query_type: str, w_sampler, n_reward_samples: int, true_delta: Optional[float] = None
) -> np.ndarray:
    w_samples, _ = w_sampler.sample_given_delta(n_reward_samples, query_type, true_delta)
    mean_weight = np.mean(w_samples, axis=0)
    normalized_mean_weight = mean_weight / np.linalg.norm(mean_weight)
    return normalized_mean_weight


def save_reward(
    query_type: str,
    w_sampler,
    n_reward_samples: int,
    outdir: Path,
    true_delta: Optional[float] = None,
):
    np.save(
        outdir / "mean_reward.npy",
        make_mode_reward(query_type, w_sampler, n_reward_samples, true_delta),
    )


def update_inputs(
    a_inputs: np.ndarray, b_inputs: np.ndarray, inputs: np.ndarray, outdir: Path
) -> np.ndarray:
    """Adds a new pair of input trajectories (a_inputs, b_inputs) to the inputs list and saves it."""
    inputs = append(inputs, np.stack([a_inputs, b_inputs]))
    np.save(outdir / "inputs.npy", inputs)
    return inputs


def append(a: Optional[np.ndarray], b: Union[np.ndarray, int], flat=False) -> np.ndarray:
    if isinstance(b, np.ndarray) and not flat:
        b = b.reshape((1, *b.shape))

    if a is None:
        if isinstance(b, np.ndarray):
            return b
        elif isinstance(b, int):
            return np.array([b])
    else:
        if isinstance(b, np.ndarray):
            return np.append(a, b, axis=0)
        elif isinstance(b, int):
            return np.append(a, b)


def setup(task: str, criterion: str, query_type: str, outdir: Path, delta: Optional[float] = None):
    assert delta is None or delta > 1.0, "Delta must be strcitly greater than 1"
    task = task.lower()
    criterion = criterion.lower()
    query_type = query_type.lower()
    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    now = datetime.now()

    logging.basicConfig(level="INFO", filename=outpath / f"{now}.log")

    assert criterion == "information" or criterion == "volume" or criterion == "random", (
        "There is no criterion called " + criterion
    )
    return task, criterion, query_type, outpath


def simulated(
    task: str,
    criterion: str,
    query_type: str,
    termination_threshold: float,
    n_reward_samples: int,
    equiv_size: Optional[float] = None,
    true_reward_path: Optional[Path] = None,
    outdir: Path = Path("questions"),
    continuous: bool = False,
    overwrite: bool = False,
    n_replications: int = 1,
):
    n_replications = int(n_replications)
    if n_replications > 1:
        Parallel(n_jobs=-2)(
            delayed(simulated)(
                task,
                criterion,
                query_type,
                termination_threshold,
                n_reward_samples,
                equiv_size,
                true_reward_path,
                Path(outdir) / str(i),
                continuous,
                overwrite,
            )
            for i in range(n_replications)
        )
        exit()

    task, criterion, query_type, outdir = setup(
        task, criterion, query_type, outdir, delta=equiv_size
    )

    simulation_object = create_env(task)
    d = simulation_object.num_of_features

    if true_reward_path is not None:
        true_reward = np.load(true_reward_path)
    else:
        true_reward = np.random.normal(size=(4,))
        true_reward = true_reward / np.linalg.norm(true_reward)
        np.save(outdir / "true_reward.npy", true_reward)

    pickle.dump(
        {
            "task": task,
            "criterion": criterion,
            "query_type": query_type,
            "epsilon": termination_threshold,
            "reward_iterations": n_reward_samples,
            "equiv_size": equiv_size,
            "continuous": continuous,
        },
        open(outdir / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outdir, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(outdir, filename="preferences.npy", overwrite=overwrite)
    inputs: np.ndarray = load(outdir, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(outdir, filename="input_features.npy", overwrite=overwrite)

    # If there is already data, feed it to the w_sampler to get the right posterior.
    w_sampler = Sampler(d)
    if inputs is not None and preferences is not None:
        for (a_phi, b_phi), preference in zip(input_features, preferences):
            w_sampler.feed(a_phi, b_phi, [preference])

    score = np.inf
    try:
        while score >= termination_threshold:
            w_samples, delta_samples = w_sampler.sample_given_delta(
                sample_count=n_reward_samples, query_type=query_type, delta=equiv_size
            )

            input_A, input_B, score = run_algo(
                criterion, simulation_object, w_samples, delta_samples, continuous
            )
            logging.info(f"Score={score}")

            if score > termination_threshold:
                inputs = update_inputs(
                    a_inputs=input_A, b_inputs=input_B, inputs=inputs, outdir=outdir
                )
                phi_A, phi_B, preference = get_simulated_feedback(
                    simulation=simulation_object,
                    input_A=input_A,
                    input_B=input_B,
                    query_type=query_type,
                    true_reward=true_reward,
                    delta=equiv_size,
                )
                input_features = append(input_features, np.stack([phi_A, phi_B]))
                normals = append(normals, phi_A - phi_B)
                preferences = append(preferences, preference)
                np.save(outdir / "input_features.npy", input_features)
                np.save(outdir / "normals.npy", normals)
                np.save(outdir / "preferences.npy", preferences)

                w_sampler.feed(phi_A, phi_B, [preference])
    except KeyboardInterrupt:
        # Pass through to finally
        logging.warning("\nSaving results, please do not exit again.")
    finally:
        save_reward(query_type, w_sampler, n_reward_samples, outdir, true_delta=equiv_size)


def human(
    task: str,
    criterion: str,
    query_type: str,
    epsilon: float,
    n_reward_samples: int,
    equiv_size: float,
    outdir: Path = Path("questions"),
    continuous: bool = False,
    overwrite: bool = False,
):
    task, criterion, query_type, outdir = setup(
        task, criterion, query_type, outdir, delta=equiv_size
    )

    simulation_object = create_env(task)
    d = simulation_object.num_of_features

    pickle.dump(
        {
            "task": task,
            "criterion": criterion,
            "query_type": query_type,
            "epsilon": epsilon,
            "reward_iterations": n_reward_samples,
            "delta": equiv_size,
            "continuous": continuous,
        },
        open(outdir / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outdir, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(outdir, filename="preferences.npy", overwrite=overwrite)
    inputs: np.ndarray = load(outdir, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(outdir, filename="input_features.npy", overwrite=overwrite)

    w_sampler = Sampler(d)
    if inputs is not None and preferences is not None:
        for (a_phi, b_phi), preference in zip(input_features, preferences):
            w_sampler.feed(a_phi, b_phi, [preference])

    score = np.inf
    try:
        while score >= epsilon:
            w_samples, delta_samples = w_sampler.sample_given_delta(
                n_reward_samples, query_type, equiv_size
            )

            input_A, input_B, score = run_algo(
                criterion, simulation_object, w_samples, delta_samples, continuous
            )

            if score > epsilon:
                inputs = update_inputs(
                    a_inputs=input_A, b_inputs=input_B, inputs=inputs, outdir=outdir
                )
                phi_A, phi_B, preference = get_feedback(
                    simulation_object, input_A, input_B, query_type
                )
                input_features = append(input_features, np.stack([phi_A, phi_B]))
                normals = append(normals, phi_A - phi_B)
                preferences = append(preferences, preference)
                np.save(outdir / "input_features.npy", input_features)
                np.save(outdir / "normals.npy", normals)
                np.save(outdir / "preferences.npy", preferences)

                w_sampler.feed(phi_A, phi_B, [preference])
    except KeyboardInterrupt:
        # Pass through to finally
        logging.warning("\nSaving results, please do not exit again.")
    finally:
        save_reward(query_type, w_sampler, n_reward_samples, outdir, true_delta=equiv_size)


if __name__ == "__main__":
    fire.Fire()
