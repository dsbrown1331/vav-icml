import logging
import pickle
from pathlib import Path

import fire  # type: ignore
import numpy as np
from scipy.linalg import norm  # type: ignore

from elicitation import append, load, save_reward, update_inputs
from sampling import Sampler
from simulation_utils import create_env, get_feedback, get_simulated_feedback


def update_response(
    input_features: np.ndarray,
    normals: np.ndarray,
    preferences: np.ndarray,
    phi_A: np.ndarray,
    phi_B: np.ndarray,
    preference: int,
    outdir: Path,
):
    input_features = append(input_features, np.stack([phi_A, phi_B]))
    normals = append(normals, phi_A - phi_B)
    preferences = append(preferences, preference)
    np.save(outdir / "input_features.npy", input_features)
    np.save(outdir / "normals.npy", normals)
    np.save(outdir / "preferences.npy", preferences)
    return input_features, normals, preferences


def make_random_questions(n_questions: int, simulation_object) -> np.ndarray:
    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]
    inputs = np.random.uniform(
        low=2 * lower_input_bound,
        high=2 * upper_input_bound,
        size=(n_questions, 2 * simulation_object.feed_size),
    )
    inputs = np.reshape(inputs, (n_questions, 2, -1))
    return inputs


def simulated(
    task: str,
    query_type: str,
    n_questions: int,
    equiv_size: float = 1.1,
    reward_iterations: int = 100,
    outdir: Path = Path("data/simulated/random/elicitation"),
    reward_path: Path = Path("data/simulated/true_reward.npy"),
    overwrite: bool = False,
) -> None:
    # TODO(AUTHOR_1): Factor most of this out to be common with human. Literally all I changed was loading true_reward and
    # changing get_feedback to get_simulated_feedback.
    logging.basicConfig(level=logging.INFO)

    outpath = Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    reward_path = Path(reward_path)
    if not reward_path.exists():
        true_reward = np.random.default_rng().normal(loc=0, scale=1, size=(4,))
        true_reward = true_reward / norm(true_reward)
        np.save(reward_path, true_reward)
    else:
        true_reward = np.load(reward_path)

    pickle.dump(
        {
            "task": task,
            "query_type": query_type,
            "n_questions": n_questions,
            "equiv_size": equiv_size,
            "reward_iterations": reward_iterations,
        },
        open(outpath / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outpath, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(outpath, filename="preferences.npy", overwrite=overwrite)
    inputs: np.ndarray = load(outpath, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(outpath, filename="input_features.npy", overwrite=overwrite)

    simulation_object = create_env(task)

    if inputs is not None and inputs.shape[0] > input_features.shape[0]:
        logging.info("Catching up.")
        input_A, input_B = inputs[-1]

        phi_A, phi_B, preference = get_simulated_feedback(
            simulation_object, input_A, input_B, query_type, true_reward, equiv_size
        )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    # Questions and inputs are duplicated, but this keeps everything consistent for the hot-load case
    new_questions = n_questions - inputs.shape[0] if inputs is not None else n_questions
    questions = make_random_questions(
        n_questions=new_questions, simulation_object=simulation_object
    )

    if inputs is not None:
        assert inputs.shape[0] == input_features.shape[0]
        assert inputs.shape[0] == normals.shape[0]
        assert inputs.shape[0] == preferences.shape[0]

    for input_A, input_B in questions:
        inputs = update_inputs(input_A, input_B, inputs, outpath)

        if inputs.shape[0] % 10 == 0:
            logging.info(f"{inputs.shape[0]} of {n_questions}")

        phi_A, phi_B, preference = get_simulated_feedback(
            simulation_object, input_A, input_B, query_type, true_reward, equiv_size
        )

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    save_reward(
        query_type=query_type,
        true_delta=equiv_size,
        w_sampler=Sampler(simulation_object.num_of_features),
        M=reward_iterations,
        outdir=outpath,
    )


def human(
    task: str,
    query_type: str,
    n_questions: int,
    delta: float = 1.1,
    reward_iterations: int = 100,
    outdir: Path = Path("random_questions"),
    overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)

    outpath = Path(outdir)

    if not outpath.exists():
        outpath.mkdir()

    pickle.dump(
        {
            "task": task,
            "query_type": query_type,
            "n_questions": n_questions,
            "delta": delta,
            "reward_iterations": reward_iterations,
        },
        open(outpath / "flags.pkl", "wb"),
    )

    normals: np.ndarray = load(outpath, filename="normals.npy", overwrite=overwrite)
    preferences: np.ndarray = load(outpath, filename="preferences.npy", overwrite=overwrite)
    inputs: np.ndarray = load(outpath, filename="inputs.npy", overwrite=overwrite)
    input_features: np.ndarray = load(outpath, filename="input_features.npy", overwrite=overwrite)

    simulation_object = create_env(task)

    if inputs is not None and inputs.shape[0] > input_features.shape[0]:
        logging.info("Catching up.")
        input_A, input_B = inputs[-1]

        phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    # Questions and inputs are duplicated, but this keeps everything consistent for the hot-load case
    questions = make_random_questions(
        n_questions=n_questions - inputs.shape[0], simulation_object=simulation_object
    )

    if inputs is not None:
        assert inputs.shape[0] == input_features.shape[0]
        assert inputs.shape[0] == normals.shape[0]
        assert inputs.shape[0] == preferences.shape[0]

    for input_A, input_B in questions:
        inputs = update_inputs(input_A, input_B, inputs, outpath)

        if inputs.shape[0] % 10 == 0:
            logging.info(f"{inputs.shape[0]} of {n_questions}")

        phi_A, phi_B, preference = get_feedback(simulation_object, input_A, input_B, query_type)

        input_features, normals, preferences = update_response(
            input_features, normals, preferences, phi_A, phi_B, preference, outpath
        )

    save_reward(
        query_type=query_type,
        true_delta=delta,
        w_sampler=Sampler(simulation_object.num_of_features),
        M=reward_iterations,
        outdir=outpath,
    )


if __name__ == "__main__":
    fire.Fire({"human": human, "simulated": simulated})
