#!/usr/bin/env python3
# @author:

import utils
import numpy as np
import copy


class Env:
    def __init__(self, width, length, obstacles=None):
        self.origin = (0, 0)
        self.width = width  # x
        self.length = length  # y
        self.boundaries = np.array(
            [
                [self.origin[0], self.origin[1]],
                [self.origin[0] + self.width, self.origin[1]],
                [self.origin[0] + self.width, self.origin[1] + self.length],
                [self.origin[0], self.origin[1] + self.length]
            ]
        )
        if obstacles is not None:
            for obstacle in obstacles:
                if not self.check_position(obstacle['origin']):
                    raise ValueError('Obstacle origin cannot be out of the environment')
            # obstacles = [{'origin':(o1_x,o1_y), 'width':o1_width, 'length':o1_length},{}]
            self.obstacles = self.coordinate_range(obstacles)

    def coordinate_range(self, obstacles):
        obstacles_w_range = copy.deepcopy(obstacles)
        for idx, obstacle in enumerate(obstacles):
            corner1 = obstacle['origin']
            # corner2 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1])
            corner3 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1] + obstacle['length'])
            # corner4 = (obstacle['origin'][0], obstacle['origin'][1] + obstacle['length'])

            if not self.check_position(corner1):
                raise ValueError('Obstacle origin cannot be out of the environment')
            range_width = [corner1[0], corner3[0]]
            range_length = [corner1[1], corner3[1]]
            obstacles_w_range[idx]['range_width'] = range_width
            obstacles_w_range[idx]['range_length'] = range_length
        return obstacles_w_range

    def check_position(self, position, check_obstacle=None):
        """
        Check if the given position falls on the boundary or an obstacle
        :param check_obstacle:
        :param position:
        :return:
        """

        if position[0] <= self.origin[0] or position[1] <= self.origin[1] \
                or position[0] >= self.width or position[1] >= self.length:
            return False

        if check_obstacle is not None:
            for obstacle in self.obstacles:
                if position[0] >= obstacle['range_width'][0] and position[0] >= obstacle['range_width'][1] \
                        and position[1] >= obstacle['range_length'][0] and position[1] >= obstacle['range_length'][1]:
                    return False
        return True


class AllAgents:
    def __init__(self):
        self.sources = {}
        self.agents = {}

    def update_agent(self, agent, agent_type='source'):
        if agent_type != 'source':
            self.agents[agent.id] = agent
        else:
            self.sources[agent.id] = agent

    def get_sensing_info(self, agent_id):
        """
        Compute sources and other agents sensing information in all quadrant
        :param agent_id:
        :return:
        """
        target_agent = self.agents[agent_id]
        # diff_info_type = len(target_agent.info_efficiency.key())
        sensing_agents_quadrants = {}  # np.zeros((diff_info_type,4)).ravel()
        sensing_sources_quadrants = {}  # np.zeros((diff_info_type, 4)).ravel()
        for info_type in target_agent.info_efficiency.keys():
            quadrants_agents = np.zeros((1, 4)).ravel()
            quadrants_sources = np.zeros((1, 4)).ravel()
            max_info_sensing_agents = target_agent.info_efficiency[info_type] * np.array((1, 4)).ravel()
            max_info_sensing_sources = target_agent.info_efficiency[info_type] * np.array((1, 4)).ravel()
            for agent in self.agents.values():
                if agent.id != target_agent.id:
                    distance_aj = utils.euclidian_distance(agent.current_position, target_agent.current_position) + 1
                    # Quadrant 0
                    if agent.current_position[0] < target_agent.current_position[0] and \
                            agent.current_position[1] < target_agent.current_position[1]:
                        quadrants_agents[0] = quadrants_agents[0] + agent.info_efficiency[info_type] / distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if agent.info_efficiency[info_type] > max_info_sensing_agents[0]:
                            max_info_sensing_agents[0] = agent.info_efficiency[info_type]

                    # Quadrant 1
                    elif agent.current_position[0] >= target_agent.current_position[0] and \
                            agent.current_position[1] < target_agent.current_position[1]:
                        quadrants_agents[1] = quadrants_agents[1] + agent.info_efficiency[info_type] / distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if agent.info_efficiency[info_type] > max_info_sensing_agents[1]:
                            max_info_sensing_agents[1] = agent.info_efficiency[info_type]

                    # Quadrant 2
                    elif agent.current_position[0] >= target_agent.current_position[0] and \
                            agent.current_position[1] >= target_agent.current_position[1]:
                        quadrants_agents[2] = quadrants_agents[2] + agent.info_efficiency[info_type] / distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if agent.info_efficiency[info_type] > max_info_sensing_agents[2]:
                            max_info_sensing_agents[2] = agent.info_efficiency[info_type]

                    # Quadrant 3
                    else:
                        quadrants_agents[3] = quadrants_agents[3] + agent.info_efficiency[info_type] / distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if agent.info_efficiency[info_type] > max_info_sensing_agents[3]:
                            max_info_sensing_agents[3] = agent.info_efficiency[info_type]

            sensing_agents_quadrants[info_type] = np.divide(quadrants_agents, max_info_sensing_agents)

            for source in self.sources.values():
                distance_ij = utils.euclidian_distance(source.position, target_agent.current_position) + 1
                # Quadrant 0
                if source.position[0] < target_agent.current_position[0] and \
                        source.position[1] < target_agent.current_position[1]:
                    quadrants_sources[0] = quadrants_sources[0] + source.info_strength[info_type] / distance_ij

                    # Check if the value of the information type for agent i is greater than max
                    if source.info_strength[info_type] > max_info_sensing_sources[0]:
                        max_info_sensing_sources[0] = source.info_strength[info_type]

                # Quadrant 1
                elif source.position[0] >= target_agent.current_position[0] and \
                        source.position[1] < target_agent.current_position[1]:
                    quadrants_sources[1] = quadrants_sources[1] + source.info_strength[info_type] / distance_ij

                    # Check if the value of the information type for agent i is greater than max
                    if source.info_strength[info_type] > max_info_sensing_sources[1]:
                        max_info_sensing_sources[1] = source.info_strength[info_type]

                # Quadrant 2
                elif source.position[0] >= target_agent.current_position[0] and \
                        source.position[1] >= target_agent.current_position[1]:
                    quadrants_sources[2] = quadrants_sources[2] + source.info_strength[info_type] / distance_ij

                    # Check if the value of the information type for agent i is greater than max
                    if source.info_strength[info_type] > max_info_sensing_sources[2]:
                        max_info_sensing_sources[2] = source.info_strength[info_type]

                # Quadrant 3
                else:
                    quadrants_sources[3] = quadrants_sources[3] + source.info_strength[info_type] / distance_ij

                    # Check if the value of the information type for agent i is greater than max
                    if source.info_strength[info_type] > max_info_sensing_sources[3]:
                        max_info_sensing_sources[3] = source.info_strength[info_type]

            sensing_sources_quadrants[info_type] = np.divide(quadrants_sources, max_info_sensing_sources)

        return sensing_agents_quadrants, sensing_sources_quadrants
