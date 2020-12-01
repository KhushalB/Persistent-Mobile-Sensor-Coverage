import numpy as np


def euclidian_distance(v1, v2):
    v1, v2 = np.array(v1).ravel(), np.array(v2).ravel()
    return np.sqrt(np.sum((v1 - v2) ** 2))


def dropoff_function(source_pos, agent_pos, a, n, lambda_w, lambda_n):
    distance = (source_pos[0] - agent_pos[0]) ** 2 + (source_pos[1] - agent_pos[1]) ** 2
    dropoff = a * n * np.exp(-(distance / lambda_w)) + a * (1 - n) * np.exp(-(distance / lambda_n))
    return dropoff

# Compute the drop off from agent's position to source's position
def likelihood(source_pos, agent_pos, amplitude, sigma):
    #dimension = X.shape[1]
    source_pos = np.array(source_pos).ravel()
    agent_pos = np.array(agent_pos).ravel()
    cov = np.array([[sigma, 0],
                    [0, sigma]])
    deviation = (source_pos - agent_pos)

    numerator = np.exp(-0.5 * np.dot(deviation, np.dot(np.linalg.inv(cov), deviation.T)))
    denominator = np.sqrt(np.linalg.det(cov)*(2 * np.pi)**2)
    return amplitude * np.divide(numerator, denominator)


def get_angle_phi(env_boundaries, agent_pos, best_quadrant):
    if best_quadrant == 0:
        d = euclidian_distance([agent_pos[0], 0], agent_pos)
        hypot = euclidian_distance(agent_pos, env_boundaries[0, :])
        alpha = np.arccos(d / hypot)
        return alpha + np.pi
    elif best_quadrant == 1:
        d = euclidian_distance([agent_pos[0], agent_pos[1]], [env_boundaries[1, :][0], agent_pos[1]])
        hypot = euclidian_distance(agent_pos, env_boundaries[1, :])
        theta = np.arccos(d / hypot)
        return theta + np.pi / 2
    elif best_quadrant == 2:
        d = euclidian_distance([agent_pos[0], agent_pos[1]], [agent_pos[0], env_boundaries[2, :][1]])
        hypot = euclidian_distance(agent_pos, env_boundaries[2, :])
        phi = np.arccos(d / hypot)
        return phi
    else:
        d = euclidian_distance([agent_pos[0], agent_pos[1]], [0, agent_pos[1]])
        hypot = euclidian_distance(agent_pos, env_boundaries[3, :])
        beta = np.arccos(d / hypot)
        return beta + (3 / 4) * np.pi


def check_if_on_obstacle(pos, obstacles=None):
    if obstacles is not None:
        for obstacle in obstacles:
            corner1 = obstacle['origin']
            # corner2 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1])
            corner3 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1] + obstacle['length'])
            # corner4 = (obstacle['origin'][0], obstacle['origin'][1] + obstacle['length'])

            range_width = [corner1[0], corner3[0]]
            range_length = [corner1[1], corner3[1]]

            if range_width[0] <= pos[0] <= range_width[1] \
                    and range_length[0] <= pos[1] <= range_length[1]:
                return False
        return True
    else:
        ValueError('Obstacles not provided!')

