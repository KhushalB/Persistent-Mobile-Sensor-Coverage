import numpy as np


def euclidian_distance(v1, v2):
    v1, v2 = np.array(v1).ravel(), np.array(v2).ravel()
    return np.sqrt(np.sum((v1 - v2) ** 2))


def dropoff_function(source_pos, agent_pos, a, n, lambda_w, lambda_n):
    distance = (source_pos[0] - agent_pos[0]) ^ 2 + (source_pos[1] - agent_pos[1]) ^ 2
    dropoff = a * n * np.exp(-(distance / lambda_w)) + a * (1 - n) * np.exp(-(distance / lambda_n))
    return dropoff


def angle_phi(env_boundaries, agent_pos, best_quadrant):
    if best_quadrant == 0:
        d = euclidian_distance([agent_pos[0], 0], env_boundaries[0, :])
        hypot = euclidian_distance(agent_pos, env_boundaries[0, :])
        alpha = np.arccos(d / hypot)
        return alpha + np.pi
    elif best_quadrant == 1:
        d = euclidian_distance([0, agent_pos[1]], env_boundaries[1, :])
        hypot = euclidian_distance(agent_pos, env_boundaries[1, :])
        theta = np.arccos(d / hypot)
        return theta + np.pi / 2
    elif best_quadrant == 2:
        d = euclidian_distance([agent_pos[0], 0], env_boundaries[3, :])
        hypot = euclidian_distance(agent_pos, env_boundaries[2, :])
        phi = np.arccos(d / hypot)
        return phi
    else:
        d = euclidian_distance([0, agent_pos[1]], env_boundaries[0, :])
        hypot = euclidian_distance(agent_pos, env_boundaries[3, :])
        beta = np.arccos(d / hypot)
        return beta + (3/4) * np.pi
