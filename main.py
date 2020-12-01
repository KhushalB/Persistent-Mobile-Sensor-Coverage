#!/usr/bin/env python3
# @author:
import numpy as np
from world import Env, AllAgents
from agent import Agent
from source import Source
from learn import learn_with_global_reward, learn_with_local_reward, learn_with_difference_reward
from utils import check_if_on_obstacle


def get_parameters():
    """
    Create dictionary of parameters needed for simulation
    :return:
    """
    parameters = {}

    # Experiment Parameters
    parameters["env_width"] = 100
    parameters["env_length"] = 100
    parameters["n_agents"] = 11
    parameters["n_sources"] = 5
    parameters["step_size"] = 0.8
    parameters["epsilon_greedy"] = 0.1

    # Test Parameters
    parameters["s_runs"] = 1  # Number of statistical runs to perform
    # parameters["new_world_config"] = 1  # 1 = Create new environment, 0 = Use existing environment
    parameters["running"] = 1  # 1 keeps visualizer from closing (use 0 for multiple stat runs)

    # Neural Network Parameters
    parameters["n_inputs"] = 16
    parameters["n_hnodes"] = 32
    parameters["n_outputs"] = 4

    # CCEA Parameters
    parameters["pop_size"] = 50
    parameters["m_rate"] = 0.1
    parameters["m_prob"] = 0.1
    parameters["epsilon"] = 0.1
    parameters["generations"] = 300
    parameters["n_elites"] = 10

    # Training Parameters
    parameters["n_steps"] = 60

    # Drop-off Parameters
    parameters["drop_a"] = -5
    parameters["drop_n"] = 0.3
    parameters["drop_lambda_w"] = 5
    parameters["drop_lambda_n"] = 2

    parameters["amplitude"] = 2
    parameters["sigma"] = 2

    objects_size = {}
    # Environment Rendering parameters
    objects_size["agent_radius"] = 1.0
    objects_size["source_radius"] = 2.0
    objects_size["border_width"] = 1

    return parameters, objects_size


if __name__ == "__main__":
    rewards = ['global_reward', 'difference_reward', 'local_reward']

    # defining obstacles
    obstacles = [
        {'origin': (65, 30), 'width': 5, 'length': 17},
        {'origin': (5, 40), 'width': 5, 'length': 7},
        {'origin': (25, 10), 'width': 3, 'length': 10},
        {'origin': (10, 60), 'width': 7, 'length': 2}
    ]

    # Get the parameters
    param, object_size = get_parameters()

    env = Env(param["env_width"], param["env_length"], object_size, obstacles)
    all_agents = AllAgents(object_size, type_reward=rewards[0])

    # Position the source withing the environment
    init_source_pos = [[5, 25], [15, 75], [65, 85], [50, 50], [80, 20]]
    for i in range(param["n_sources"]):
        info_strength = {'precipitation': 10, 'temperature': 20}
        # x0, y0 = np.random.uniform(low=0.0, high=param["env_width"]), np.random.uniform(low=0.0, high=param["env_width"])
        # ini_position = [x0, y0]
        ini_position = init_source_pos[i]
        source = Source(source_id=i, position=ini_position, info_strength=info_strength)
        all_agents.add_agent(source, agent_type='source')

    try:
        initial_pos_agents = np.loadtxt('initial_position_agents.csv', delimiter=',')
        if param["n_agents"] != len(initial_pos_agents):
            raise ValueError("Number of initial position in the file doesn't match the defined number of agents!")
        for i, ini_position in enumerate(initial_pos_agents):
            info_efficiency = {'precipitation': 3, 'temperature': 5}
            agent = Agent(param, agent_id=i, initial_position=list(ini_position), info_efficiency=info_efficiency)
            all_agents.add_agent(agent, agent_type='agent')
    except:
        initial_pos_agents = []
        for i in range(param["n_agents"]):
            info_efficiency = {'precipitation': 3, 'temperature': 5}
            x0 = np.random.uniform(
                low=0.0 + (object_size["border_width"] + object_size["agent_radius"]),
                high=param["env_width"] - (object_size["border_width"] + object_size["agent_radius"])
            )
            y0 = np.random.uniform(
                low=0.0 + (object_size["border_width"] + object_size["agent_radius"]),
                high=param["env_length"] - (object_size["border_width"] + object_size["agent_radius"])
            )
            # Check for appropriate random position for agents (out of obstacles)
            while not check_if_on_obstacle([x0, y0], obstacles) and not env.check_position([x0, y0]):
                x0 = np.random.uniform(low=0.0, high=param["env_width"])
                y0 = np.random.uniform(low=0.0, high=param["env_length"])
            # print(f'good position found for agent: {i}') #------------------------------------------------------------
            ini_position = [x0, y0]
            agent = Agent(param, agent_id=i, initial_position=ini_position, info_efficiency=info_efficiency)
            all_agents.add_agent(agent, agent_type='agent')
            initial_pos_agents.append(ini_position)
        np.savetxt('initial_position_agents.csv', initial_pos_agents, delimiter=',')

    if all_agents.type_reward == 'difference_reward':
        learn_with_difference_reward(env, param, all_agents)
    elif all_agents.type_reward == 'local_reward':
        learn_with_local_reward(env, param, all_agents)
    else:
        learn_with_global_reward(env, param, all_agents)

