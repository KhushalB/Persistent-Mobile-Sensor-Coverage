#!/usr/bin/env python3
# @author:
import numpy as np
from utils import dropoff_function, angle_phi


class Agent:
    def __init__(self, agent_id, step_size, initial_position, info_efficiency=None):
        if info_efficiency in None:
            info_efficiency = {'precipitation': 5, 'temperature': 3}
        self.id = agent_id
        self.info_efficiency = info_efficiency
        self.current_position = initial_position  # 2d vector containing x0 and y0
        self.step_size = step_size
        self.quadrant_value_table = np.random.uniform(0, 1, (1, 4))
        # diff_info_type = len(self.info_efficiency.keys())
        self.sensing_of_sources, self.sensing_other_agents = {}, {}
        for info_type in self.info_efficiency.keys():
            self.sensing_of_sources[info_type] = np.array((1, 4)).ravel()  # Sensing in quadrant 0, 1, 2 and 3
            self.sensing_other_agents[info_type] = np.array((1, 4)).ravel()  # Sensing in quadrant 0, 1, 2 and 3

    def move(self, phi):
        """
        Compute the angle for movement direction
        :param phi:
        :return: new position of the agent after moving one step
        """
        new_x = self.current_position[0] + self.step_size * np.cos(phi)  # np.cos(phi * np.pi / 180)
        new_y = self.current_position[1] + self.step_size * np.sin(phi)  # np.sin(phi * np.pi / 180)
        self.current_position = [new_x, new_y]
        return self.current_position
