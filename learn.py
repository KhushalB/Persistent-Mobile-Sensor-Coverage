#!/usr/bin/env python3
# @author:
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from evolutionary_algo import EvolutionaryAlgorithm as EA
from world import AllAgents
from utils import get_angle_phi
import csv


def save_reward_history(reward_history, file_name):
    """
    Saves the reward history of the agent teams to create plots for learning performance
    :param reward_history:
    :param file_name:
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'a+', newline='') as csvfile:  # Record reward history for each stat run
        writer = csv.writer(csvfile)
        writer.writerow(['Performance'] + reward_history)


def save_path_history(path_dictionary, file_name):
    """
    Save the path of all the agents from the last learning episode
    :param path_dictionary:
    :param file_name:
    :return:
    """
    dir_name = 'Output_Data/'  # Intended directory for output files
    save_file_name = os.path.join(dir_name, file_name)

    with open(save_file_name, 'w') as fp:
        json.dump(path_dictionary, fp)


def learn_with_global_reward(env, parameters, all_agents):
    """
    Train the agents using the global reward
    :param reward_type:
    :return:
    """
    p = parameters  # get_parameters()
    # rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    sensor_agent = {}
    for agent in all_agents.agents.values():
        sensor_agent["EA{0}".format(agent.id)] = EA(p)

    # print("Reward Type: ", reward_type)
    # print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for agent in all_agents.agents.values():
            sensor_agent["EA{0}".format(agent.id)].create_new_population()
        reward_history = []

        for gen in tqdm(range(p["generations"])):
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                # rd.clear_rover_path()
                for agent in all_agents.agents.values():
                    all_agents.agents[agent.id].reset_agent()  # Reset agent to initial conditions
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                    all_agents.agents[agent.id].get_network_weights(weights)  # Apply network weights from CCEA
                    # rd.update_rover_path(sensor_agent["AG{0}".format(agent.id)], agent.id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for agent in all_agents.agents.values():
                        # Get sensing of sources and other agents
                        a_it, s_it = all_agents.get_sensing_info(agent.id)
                        agent.get_inputs(a_it, s_it)
                        agent.get_outputs()
                        best_quadrant = np.argmax(agent.output_layer)
                        angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                        agent.step(angle_phi, env, all_agents)

                # Update fitness of policies using reward information
                all_agents.compute_source_coverage()
                #all_agents.compute_source_coverage(p["drop_a"], p["drop_n"], p["drop_lambda_w"], p["drop_lambda_n"])
                #all_agents.compute_source_coverage(p["amplitude"], p["sigma"])
                global_reward = all_agents.global_fitness()
                for agent in all_agents.agents.values():
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    sensor_agent["EA{0}".format(agent.id)].fitness[policy_id] = global_reward

            # Testing Phase (test best policies found so far)
            # rd.clear_rover_path()
            for agent in all_agents.agents.values():
                all_agents.agents[agent.id].reset_agent()  # Reset rover to initial conditions
                policy_id = np.argmax(sensor_agent["EA{0}".format(agent.id)].fitness)
                weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                all_agents.agents[agent.id].get_network_weights(weights)  # Apply best set of weights to network
                # rd.update_rover_path(sensor_agent["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment and constructs state vector
                for agent in all_agents.agents.values():
                    # Get sensing of sources and other agents
                    a_it, s_it = all_agents.get_sensing_info(agent.id)
                    agent.get_inputs(a_it, s_it)
                    agent.get_outputs()
                    # select a quadrant at random for exploration with probability < epsilon_greedy
                    prob = np.random.uniform(0, 1, 1)
                    if prob < p["epsilon_greedy"]:
                        best_quadrant = np.random.randint(0, 4, 1)[0]
                    else:
                        best_quadrant = np.argmax(agent.output_layer)
                    angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                    #print('Shape output layer:', agent.output_layer.shape)
                    #print('output layer:', agent.output_layer)
                    #print('best quadrant:', best_quadrant, 'angle:', angle_phi) #--------------------------------------
                    agent.step(angle_phi, env, all_agents)

                    # update the agents' path if last generation
                    if gen == (p["generations"] - 1):
                        agent.update_path()
                if gen == (p["generations"] - 1):
                    env.render(all_agents.sources, all_agents.agents, filename=f'sys_state_{step_id}')

                    #if step_id % 15 == 0:
                    #    for source in all_agents.sources.values():
                    #        source.move(env, all_agents.agents)

            # Update fitness of policies using reward information
            global_reward = all_agents.global_fitness()
            reward_history.append(global_reward)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                # Save the path of the agents
                all_agents_paths = {}
                for agent in all_agents.agents.values():
                    all_agents_paths[agent.id] = agent.path
                save_path_history(all_agents_paths, "AllAgentsPath_GlobalReward.json")

            # Choose new parents and create new offspring population
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].down_select()

        plt.plot(reward_history)
        plt.title("Coverage performance under global reward")
        plt.xlabel("network generation")
        plt.ylabel("G(z)")
        plt.show()
        save_reward_history(reward_history, "Global_Reward.csv")

    # run_visualizer(p)


def learn_with_difference_reward(env, parameters, all_agents):
    """
    Train the agents using the global reward
    :param all_agents:
    :param parameters:
    :param env:
    :param reward_type:
    :return:
    """
    p = parameters  # get_parameters()
    # rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    sensor_agent = {}
    for agent in all_agents.agents.values():
        sensor_agent["EA{0}".format(agent.id)] = EA(p)

    # print("Reward Type: ", reward_type)
    # print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for agent in all_agents.agents.values():
            sensor_agent["EA{0}".format(agent.id)].create_new_population()
        reward_history = []

        for gen in tqdm(range(p["generations"])):
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                # rd.clear_rover_path()
                for agent in all_agents.agents.values():
                    all_agents.agents[agent.id].reset_agent()  # Reset agent to initial conditions
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                    all_agents.agents[agent.id].get_network_weights(weights)  # Apply network weights from CCEA
                    # rd.update_rover_path(sensor_agent["AG{0}".format(agent.id)], agent.id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for agent in all_agents.agents.values():
                        # Get sensing of sources and other agents
                        a_it, s_it = all_agents.get_sensing_info(agent.id)
                        agent.get_inputs(a_it, s_it)
                        agent.get_outputs()
                        best_quadrant = np.argmax(agent.output_layer)
                        angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                        agent.step(angle_phi, env, all_agents)

                # Update fitness of policies using reward information
                all_agents.compute_source_coverage()
                for agent in all_agents.agents.values():
                    difference_reward = agent.difference_reward(p, all_agents)
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    sensor_agent["EA{0}".format(agent.id)].fitness[policy_id] = difference_reward

            # Testing Phase (test best policies found so far)
            # rd.clear_rover_path()
            for agent in all_agents.agents.values():
                all_agents.agents[agent.id].reset_agent()  # Reset rover to initial conditions
                policy_id = np.argmax(sensor_agent["EA{0}".format(agent.id)].fitness)
                weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                all_agents.agents[agent.id].get_network_weights(weights)  # Apply best set of weights to network
                # rd.update_rover_path(sensor_agent["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Agent scans quadrants and build state vector
                for agent in all_agents.agents.values():
                    # Get sensing of sources and other agents
                    a_it, s_it = all_agents.get_sensing_info(agent.id)
                    agent.get_inputs(a_it, s_it)
                    agent.get_outputs()

                    # select a quadrant at random for exploration with probability < epsilon_greedy
                    prob = np.random.uniform(0, 1, 1)
                    if prob < p["epsilon_greedy"]:
                        best_quadrant = np.random.randint(0, 4, 1)
                    else:
                        best_quadrant = np.argmax(agent.output_layer)
                    angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                    agent.step(angle_phi, env, all_agents)

                    # update the agents' path if last generation
                    if gen == (p["generations"] - 1):
                        agent.update_path()

                if gen == (p["generations"] - 1):
                    env.render(all_agents.sources, all_agents.agents, filename=f'sys_state_{step_id}')

                    #if step_id % 15 == 0:
                    #    for source in all_agents.sources.values():
                    #        source.move(env, all_agents.agents)

            # Update fitness of policies using reward information
            system_performance = all_agents.global_fitness()
            reward_history.append(system_performance)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                # Save the path of the agents
                all_agents_paths = {}
                for agent in all_agents.agents.values():
                    all_agents_paths[agent.id] = agent.path
                save_path_history(all_agents_paths, "AllAgentsPath_DifferenceReward.json")

            # Choose new parents and create new offspring population
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].down_select()

        plt.plot(reward_history)
        plt.title("Coverage performance under difference reward")
        plt.xlabel("network generation")
        plt.ylabel("G(z)")
        plt.show()
        save_reward_history(reward_history, "Difference_Reward.csv")


def learn_with_local_reward(env, parameters, all_agents):
    """
    Train the agents using the global reward
    :param all_agents:
    :param parameters:
    :param env:
    :param reward_type:
    :return:
    """
    p = parameters  # get_parameters()
    # rd = RoverDomain(p)

    # Create dictionary for each instance of rover and corresponding NN and EA population
    sensor_agent = {}
    for agent in all_agents.agents.values():
        sensor_agent["EA{0}".format(agent.id)] = EA(p)

    # print("Reward Type: ", reward_type)
    # print("Coupling Requirement: ", p["c_req"])

    for srun in range(p["s_runs"]):  # Perform statistical runs
        print("Run: %i" % srun)

        # Reset CCEA and NN new stat run
        for agent in all_agents.agents.values():
            sensor_agent["EA{0}".format(agent.id)].create_new_population()
        reward_history = []

        for gen in tqdm(range(p["generations"])):
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].select_policy_teams()
            for team_number in range(p["pop_size"]):  # Each policy in CCEA is tested in teams
                # rd.clear_rover_path()
                for agent in all_agents.agents.values():
                    all_agents.agents[agent.id].reset_agent()  # Reset agent to initial conditions
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                    all_agents.agents[agent.id].get_network_weights(weights)  # Apply network weights from CCEA
                    # rd.update_rover_path(sensor_agent["AG{0}".format(agent.id)], agent.id, -1)  # Record starting position of each rover

                for step_id in range(p["n_steps"]):
                    # Rover scans environment and constructs state vector
                    for agent in all_agents.agents.values():
                        # Get sensing of sources and other agents
                        a_it, s_it = all_agents.get_sensing_info(agent.id)
                        agent.get_inputs(a_it, s_it)
                        agent.get_outputs()
                        best_quadrant = np.argmax(agent.output_layer)
                        angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                        agent.step(angle_phi, env, all_agents)

                # Update fitness of policies using reward information
                all_agents.compute_source_coverage()
                for agent in all_agents.agents.values():
                    local_reward = agent.local_reward(p, all_agents)
                    policy_id = int(sensor_agent["EA{0}".format(agent.id)].team_selection[team_number])
                    sensor_agent["EA{0}".format(agent.id)].fitness[policy_id] = local_reward

            # Testing Phase (test best policies found so far)
            # rd.clear_rover_path()
            for agent in all_agents.agents.values():
                all_agents.agents[agent.id].reset_agent()  # Reset rover to initial conditions
                policy_id = np.argmax(sensor_agent["EA{0}".format(agent.id)].fitness)
                weights = sensor_agent["EA{0}".format(agent.id)].population["pop{0}".format(policy_id)]
                all_agents.agents[agent.id].get_network_weights(weights)  # Apply best set of weights to network
                # rd.update_rover_path(sensor_agent["AG{0}".format(rv_id)], rv_id, -1)

            for step_id in range(p["n_steps"]):
                # Rover scans environment and constructs state vector
                for agent in all_agents.agents.values():
                    # Get sensing of sources and other agents
                    a_it, s_it = all_agents.get_sensing_info(agent.id)
                    agent.get_inputs(a_it, s_it)
                    agent.get_outputs()

                    # select a quadrant at random for exploration with probability < epsilon_greedy
                    prob = np.random.uniform(0, 1, 1)
                    if prob < p["epsilon_greedy"]:
                        best_quadrant = np.random.randint(0, 4, 1)
                    else:
                        best_quadrant = np.argmax(agent.output_layer)
                    angle_phi = get_angle_phi(env.boundaries, agent.current_position, best_quadrant)
                    agent.step(angle_phi, env, all_agents)

                    # update the agents' path if last generation
                    if gen == (p["generations"] - 1):
                        agent.update_path()
                if gen == (p["generations"] - 1):
                    env.render(all_agents.sources, all_agents.agents, filename=f'sys_state_{step_id}')

                    #if step_id % 15 == 0:
                    #    for source in all_agents.sources.values():
                    #        source.move(env, all_agents.agents)

            # Update fitness of policies using reward information
            system_performance = all_agents.global_fitness()
            reward_history.append(system_performance)

            if gen == (p["generations"] - 1):  # Save path at end of final generation
                # Save the path of the agents
                all_agents_paths = {}
                for agent in all_agents.agents.values():
                    all_agents_paths[agent.id] = agent.path
                save_path_history(all_agents_paths, "AllAgentsPath_LocalReward.json")

            # Choose new parents and create new offspring population
            for agent in all_agents.agents.values():
                sensor_agent["EA{0}".format(agent.id)].down_select()

        plt.plot(reward_history)
        plt.title("Coverage performance under local reward")
        plt.xlabel("network generation")
        plt.ylabel("G(z)")
        plt.show()
        save_reward_history(reward_history, "Local_Reward.csv")
