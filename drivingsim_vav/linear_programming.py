# from cvxopt import matrix, solvers
import logging
import sys
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog  # type: ignore


# TODO: at some point should probably use something better than scipy, do we have a license for
# ibm's cplex solver?
def is_redundant_constraint(
    halfspace: np.ndarray, halfspaces: np.ndarray, epsilon=0.0001
) -> bool:
    # Let h be a halfspace constraint in the set of contraints H.
    # We have a constraint c^w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to
    # add c back into H.
    # Thus, we want to minimize c^T w subject to Hw >= 0.
    # First we need to change this into the form min c^T x subject to Ax <= b.
    # Our problem is equivalent to min c^T w subject to  -H w <= 0.
    halfspaces = np.array(halfspaces)
    m, _ = halfspaces.shape

    b = np.zeros(m)
    sol = linprog(
        halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1), method="revised simplex"
    )
    logging.debug(f"LP Solution={sol}")
    if sol["status"] != 0:
        logging.info("Revised simplex method failed. Trying interior point method.")
        sol = linprog(halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1))

    if (
        sol["status"] != 0
    ):  # Not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue.
        logging.error("LP NOT SOLVABLE")
        sys.exit()
    elif (
        sol["fun"] < -epsilon
    ):  # If less than zero then constraint is needed to keep c^T w >=0
        return False
    else:  # redundant since without constraint c^T w >=0
        logging.debug("Redundant")
        return True


def remove_redundant_constraints(
    halfspaces, epsilon=0.0001
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a new array with all redundant halfspaces removed.

    Parameters
    -----------
    halfspaces : list of halfspace normal vectors such that np.dot(halfspaces[i], w) >= 0 for all i

    epsilon : numerical precision for determining if redundant via LP solution 

    Returns
    -----------
    list of non-redundant halfspaces 
    """
    # for each row in halfspaces, check if it is redundant

    non_redundant_halfspaces: List[np.ndarray] = list()
    indices: List[int] = list()

    halfspaces_to_check = halfspaces

    for i, halfspace in enumerate(halfspaces):
        logging.debug(f"Checking half space {halfspace}")

        halfspaces_lp = np.array(
            [halfspace for halfspace in non_redundant_halfspaces]
            + [halfspace for halfspace in halfspaces_to_check[1:]]
        )

        # Debugging
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            logging.debug(f"halfspace_lp={halfspaces_lp}")

            num_vars = len(halfspaces[0])  # size of weight vector
            logging.debug(
                f"rank={np.linalg.matrix_rank(halfspaces_lp)} vs num_vars={num_vars}"
            )
            u, s, v = np.linalg.svd(halfspaces_lp)
            logging.debug(f"SVG s={s}")
            if (
                np.linalg.matrix_rank(halfspaces_lp) < num_vars
            ):  # check to make sure LP is well-posed
                # all remaining halfspaces are required
                # TODO: Check, but I think this is true since we first normalize and remove redundancies
                non_redundant_halfspaces.extend(halfspaces_to_check)
                indices.extend(range(i, i + len(halfspaces_to_check)))
                break

        if halfspaces_lp.shape[0] == 0 or not is_redundant_constraint(
            halfspace, halfspaces_lp, epsilon
        ):
            # keep h
            logging.debug("Not redundant")
            non_redundant_halfspaces.append(halfspace)
            indices.append(i)
        else:
            logging.debug("Redundant")

        halfspaces_to_check = halfspaces_to_check[1:]
    return np.array(non_redundant_halfspaces), np.array(indices)
