""" Post-process noise and consistency filtering. """

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import distance  # type: ignore

from linear_programming import remove_redundant_constraints
from sampling import Sampler
from simulation_utils import create_env


class TestFactory:
    def __init__(
        self,
        query_type: str,
        reward_dimension: int,
        equiv_probability: Optional[float] = None,
        deterministic: bool = False,
        n_reward_samples: Optional[int] = None,
        use_mean_reward: bool = False,
        skip_noise_filtering: bool = False,
        skip_epsilon_filtering: bool = False,
        skip_redundancy_filtering: bool = False,
    ) -> None:
        """Creates a new test factory, filtering test questions.

        Args:
            query_type (str): "weak" or "strict" for allowing or disallowing equiv preferences.

            reward_dimension (int): Dimension of the reward vector.
            equiv_probability (Optional[float], optional): If weak queries allowed, what is the
                                                           relative likelihood of sampling an equiv
                                                           preference. See Sampler for details.
                                                           Defaults to None.
            deterministic (bool, optional): Whether to use a determinstic set of rewards for
                                            debugging. If True, you must provide rewards to the
                                            filter_halfplanes method. Defaults to False.
            n_reward_samples (Optional[int], optional): How many rewards to sample when computing
                                                        the posterior. Ignored if determinstic is
                                                        True. Defaults to None.
            skip_noise_filtering (bool, optional): Skips the noise filtering step. Defaults to False.
            skip_epsilon_filtering (bool, optional): Skips the epsilon-delta filtering step.
                                                     Defaults to False.
            skip_redundancy_filtering (bool, optional): Skips the redundancy filtering step.
                                                        Defaults to False.
        """
        self.query_type = query_type
        self.n_reward_samples = n_reward_samples
        self.reward_dimension = reward_dimension
        self.equiv_probability = equiv_probability
        self.deterministic = deterministic
        self.use_mean_reward = use_mean_reward
        self.skip_noise_filtering = skip_noise_filtering
        self.skip_epsilon_filtering = skip_epsilon_filtering
        self.skip_redundancy_filtering = skip_redundancy_filtering

    def sample_rewards(
        self, a_phis: np.ndarray, b_phis: np.ndarray, preferences: np.ndarray,
    ) -> np.ndarray:
        """ Samples n_samples rewards via MCMC. """
        w_sampler = Sampler(self.reward_dimension)
        for a_phi, b_phi, preference in zip(a_phis, b_phis, preferences):
            w_sampler.feed(a_phi, b_phi, [preference])
        rewards, _ = w_sampler.sample_given_delta(
            self.n_reward_samples, self.query_type, self.equiv_probability
        )
        return rewards

    @staticmethod
    def remove_duplicates(normals: np.ndarray, precision=0.0001) -> Tuple[np.ndarray, np.ndarray]:
        """ Remove halfspaces that have small cosine similarity to another. """
        out: List[np.ndarray] = list()
        out_indices: List[int] = list()

        # Remove exact duplicates
        _, indices = np.unique(normals, return_index=True, axis=0)

        for i, normal in enumerate(normals[indices]):
            for accepted_normal in out:
                if distance.cosine(normal, accepted_normal) < precision:
                    break
            out.append(normal)
            out_indices.append(indices[i])
        return np.array(out).reshape(-1, normals.shape[1]), np.array(indices, dtype=int)

    @staticmethod
    def filter_noise(
        normals: np.ndarray,
        filtered_normals: np.ndarray,
        indices: np.ndarray,
        rewards: np.ndarray,
        noise_threshold: float,
    ):
        filtered_indices = (
            np.mean(np.dot(rewards, filtered_normals.T) > 0, axis=0) > noise_threshold
        )
        indices = indices[filtered_indices]
        assert all([row in filtered_normals for row in normals[indices]])
        filtered_normals = normals[indices].reshape(-1, normals.shape[1])
        return filtered_normals, indices

    def margin_filter(
        self,
        normals: np.ndarray,
        filtered_normals: np.ndarray,
        indices: np.ndarray,
        rewards: np.ndarray,
        epsilon: float,
        delta: Optional[float] = None,
    ):
        if self.use_mean_reward:
            reward = np.mean(rewards, axis=0)
            filtered_indices = filtered_normals @ reward > epsilon
        elif delta is not None:
            opinions = np.dot(rewards, filtered_normals.T).T
            correct_opinions = opinions > epsilon
            # Filter halfspaces that don't have 1-d probability that the expected return gap is epsilon.
            filtered_indices = np.mean(correct_opinions, axis=1) > 1.0 - delta
        else:
            raise ValueError("Must provide delta if not using point reward.")

        indices = indices[filtered_indices]
        assert all([row in filtered_normals for row in normals[indices]])
        filtered_normals = normals[indices].reshape(-1, normals.shape[1])

        return filtered_normals, indices

    def filter_halfplanes(
        self,
        inputs_features: np.ndarray,
        normals: np.ndarray,
        preferences: np.ndarray,
        epsilon: float,
        noise_threshold: Optional[float] = None,
        delta: Optional[float] = None,
        rewards: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filters test questions by removing noise answers, requiring answers have a gap of at
        least epsilon, and removing redundant questions via linear programming."""
        a_phis = inputs_features[:, 0]
        b_phis = inputs_features[:, 1]
        filtered_normals = normals
        indices = np.array(range(filtered_normals.shape[0]))

        if not self.skip_noise_filtering and noise_threshold is not None:
            if rewards is None:
                if self.deterministic:
                    raise ValueError("Must provide rewards to use deterministic mode.")
                if self.n_reward_samples is None:
                    raise ValueError("Must provide n_reward_samples if reward is not provided")

                rewards = self.sample_rewards(a_phis=a_phis, b_phis=b_phis, preferences=preferences)

            filtered_normals, indices = self.filter_noise(
                normals, filtered_normals, indices, rewards, noise_threshold
            )

            logging.info(f"After noise filtering there are {len(indices)} questions.")

        if not self.skip_epsilon_filtering and filtered_normals.shape[0] > 0:
            if not self.deterministic and self.n_reward_samples is not None:
                # This reward generation logic is jank.
                rewards = self.sample_rewards(a_phis=a_phis, b_phis=b_phis, preferences=preferences)

            filtered_normals, indices = self.margin_filter(
                normals, filtered_normals, indices, rewards, epsilon, delta
            )
            logging.info(f"After epsilon delta filtering there are {len(indices)} questions.")

        if not self.skip_redundancy_filtering and filtered_normals.shape[0] > 1:
            # Remove redundant halfspaces
            filtered_normals, constraint_indices = remove_redundant_constraints(filtered_normals)

            constraint_indices = np.array(constraint_indices, dtype=np.int)
            indices = indices[constraint_indices]
            assert np.all(normals[indices] == filtered_normals)

            logging.info(f"After removing redundancies there are {len(indices)} questions.")

        return filtered_normals, indices
