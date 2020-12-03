#!/usr/bin/env python3
# @author:
import numpy as np
from utils import dropoff_function, get_angle_phi


def softmax(inp):  # Softmax function
    """
    compute the softmax of the logit function
    :param inp: Vector of logit
    :return: softmax output
    """
    return np.exp(inp) / np.sum(np.exp(inp))


def relu(inp):  # ReLu function as activation function
    """
    ReLu neural network activation function
    :param inp: Node value before activation
    :return: Node value after activation
    """
    return np.max(inp, 0)


class Agent:
    def __init__(self, p, agent_id, initial_position, info_efficiency=None):
        if info_efficiency is None:
            info_efficiency = {'precipitation': 5, 'temperature': 3}
        self.id = agent_id
        self.info_efficiency = info_efficiency
        self.initial_position = initial_position
        self.current_position = initial_position  # 2d vector containing x0 and y0
        self.step_size = p["step_size"]
        self.quadrant_value_table = np.random.uniform(0, 1, (1, 4))
        self.path = [list(self.initial_position)]
        self.sensing_of_sources, self.sensing_other_agents = {}, {}
        for info_type in self.info_efficiency.keys():
            self.sensing_of_sources[info_type] = np.array((1, 4)).ravel()  # Sensing in quadrant 0, 1, 2 and 3
            self.sensing_other_agents[info_type] = np.array((1, 4)).ravel()  # Sensing in quadrant 0, 1, 2 and 3

        # Agent Neuro-Controller Parameters
        self.n_inputs = p["n_inputs"]
        self.n_outputs = p["n_outputs"]
        self.n_hnodes = p["n_hnodes"]  # Number of nodes in hidden layer
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])


    # compute the difference reward
    def difference_reward(self, p, all_agents):
        all_agents.compute_source_coverage_without_agent_i(agent_id=self.id)

        performance = 0
        performance_me_off = 0
        normalizer = 0
        for source in all_agents.sources.values():
            n_info_type_source = len(all_agents.sources_coverage[source.id].values())
            performance = performance + sum(all_agents.sources_coverage[source.id].values())
            normalizer = normalizer + n_info_type_source + 1
            global_performance = performance / normalizer

            performance_me_off = performance_me_off + sum(
                all_agents.sources_coverage_without_me[source.id].values()
            )
            global_performance_me_off = performance_me_off / normalizer
        return global_performance - global_performance_me_off

    def local_reward(self, p, all_agents):
        all_agents.compute_source_coverage_only_agent_i(agent_id=self.id)

        performance = 0
        normalizer = 0
        for source in all_agents.sources.values():
            n_info_type_source = len(all_agents.sources_coverage_only_me[source.id].values())
            performance = performance + sum(all_agents.sources_coverage_only_me[source.id].values())
            normalizer = normalizer + n_info_type_source + 1
        return performance / normalizer

    def reset_agent(self):
        """
        Resets the agent to its initial starting conditions (used to reset each training episode)
        :return:
        """
        # Return rover to initial configuration
        self.current_position = self.initial_position

        # Reset Neural Network
        self.weights = {}
        self.input_layer = np.reshape(np.mat(np.zeros(self.n_inputs)), [self.n_inputs, 1])
        self.hidden_layer = np.reshape(np.mat(np.zeros(self.n_hnodes)), [self.n_hnodes, 1])
        self.output_layer = np.reshape(np.mat(np.zeros(self.n_outputs)), [self.n_outputs, 1])

    def step(self, phi, env, all_agents):
        """
        Compute the angle for movement direction
        :param all_agents: object containing all agents position and states (used to check collision with self)
        :param phi:
        :return: new position of the agent after moving one step
        """
        #new_x = self.current_position[0] + self.step_size * np.cos(phi)  # np.cos(phi * np.pi / 180)
        #new_y = self.current_position[1] + self.step_size * np.sin(phi)  # np.sin(phi * np.pi / 180)
        new_x = self.current_position[0] + self.step_size * np.sin(phi)  # np.cos(phi * np.pi / 180)
        new_y = self.current_position[1] + self.step_size * np.cos(phi)  # np.sin(phi * np.pi / 180)

        # Check if the new position doesn't fall out of the environment or collide with other agents/sources
        if env.check_position([new_x, new_y]) and all_agents.check_position_sources([new_x, new_y], self.id):
            self.current_position = [new_x, new_y]
        """
        print('agent:', self.id, 'Phi:', phi)
        if not env.check_position([new_x, new_y]) or not all_agents.check_position_sources([new_x, new_y], self.id):
            best_quadrant = np.argsort(-self.output_layer, axis=0)
            second_best_phi = get_angle_phi(env.boundaries, self.current_position, best_quadrant[1])
            new_phi = second_best_phi
            print('agent:', self.id, '*****Phi second:', second_best_phi)
            if phi == second_best_phi:
                third_best_phi = get_angle_phi(env.boundaries, self.current_position, best_quadrant[2])
                new_phi = third_best_phi
                print('agent:', self.id, '---------Phi third:', third_best_phi)
            # print('position to land:', [new_x, new_y])
            # new_phi = np.random.uniform(0, 2*np.pi)
            self.step(new_phi, env, all_agents)
        else:
            self.current_position = [new_x, new_y]
        """
    def update_path(self):
        self.path.append(self.current_position)

    def get_inputs(self, a_it, s_it):  # Get inputs from state-vector
        """
        Transfer state information to the neuro-controller
        Build the state vector with sensing information of sources and other agents in each quadrant
        :return:
        """

        inp = []
        for info_type in sorted(a_it.keys()):
            for sensing_a_q in a_it[info_type]:
                inp.append(sensing_a_q)

        for info_type in sorted(s_it.keys()):
            # self.input_layer[i, 0] = self.sensor_readings[i]
            for sensing_s_q in s_it[info_type]:
                inp.append(sensing_s_q)

        self.input_layer = np.array(inp).reshape(-1,1)

    def get_network_weights(self, nn_weights):
        """
        Apply chosen network weights to the agent's neuro-controller
        :param nn_weights: Dictionary of network weights received from the CCEA
        :return:
        """
        self.weights["Layer1"] = np.reshape(np.mat(nn_weights["L1"]), [self.n_hnodes, self.n_inputs])
        self.weights["Layer2"] = np.reshape(np.mat(nn_weights["L2"]), [self.n_outputs, self.n_hnodes])
        self.weights["input_bias"] = np.reshape(np.mat(nn_weights["b1"]), [self.n_hnodes, 1])
        self.weights["hidden_bias"] = np.reshape(np.mat(nn_weights["b2"]), [self.n_outputs, 1])

    def get_outputs(self):
        """
        Run NN to generate outputs
        :return:
        """
        self.hidden_layer = np.dot(self.weights["Layer1"], self.input_layer) + self.weights["input_bias"]
        for i in range(self.n_hnodes):
            self.hidden_layer[i, 0] = relu(self.hidden_layer[i, 0])

        self.output_layer = np.dot(self.weights["Layer2"], self.hidden_layer) + self.weights["hidden_bias"]
        self.output_layer = softmax(self.output_layer)

        '''
        print('Shape output layer after:', self.output_layer.shape)
        print('Shape of input matrix:', self.input_layer.shape)
        print('Shape of weights[Layer1] matrix:', self.weights['Layer1'].shape)
        print('Shape of hidden matrix:', self.hidden_layer.shape)
        print('Shape of weights[Layer2] matrix:', self.weights['Layer2'].shape)
        print('\n \n \n \n')
        '''