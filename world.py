#!/usr/bin/env python3
# @author:
import os
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy


class Env:
    def __init__(self, width, length, obj_size, obstacles=None):
        self.origin = (0, 0)
        self.width = width  # x
        self.length = length  # y
        self.obj_size = obj_size
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
        else:
            self.obstacles = None

    def coordinate_range(self, obstacles):
        obstacles_dim_range = copy.deepcopy(obstacles)
        for idx, obstacle in enumerate(obstacles):
            corner1 = obstacle['origin']
            # corner2 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1])
            corner3 = (obstacle['origin'][0] + obstacle['width'], obstacle['origin'][1] + obstacle['length'])
            # corner4 = (obstacle['origin'][0], obstacle['origin'][1] + obstacle['length'])

            if not self.check_position(corner1):
                raise ValueError('Obstacle origin cannot be out of the environment')
            range_width = [corner1[0], corner3[0]]
            range_length = [corner1[1], corner3[1]]
            obstacles_dim_range[idx]['range_width'] = range_width
            obstacles_dim_range[idx]['range_length'] = range_length
        return obstacles_dim_range

    def check_position__(self, position, check_obstacle=None):
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
                if obstacle['range_width'][0] and position[0] >= obstacle['range_width'][1] \
                        and position[1] >= obstacle['range_length'][0] and position[1] >= obstacle['range_length'][1]:
                    return False
        return True

    def check_position(self, position):
        """
        Check if the given position falls on the boundary or an obstacle
        :param check_obstacle:
        :param position:
        :return:
        """
        agt_radius = self.obj_size['agent_radius']
        src_radius = self.obj_size['source_radius']
        env_border = self.obj_size['border_width']
        # check if the position is out of the boundaries of the environment
        if position[0] <= self.origin[0] + (env_border + agt_radius) or position[1] <= self.origin[1] + (
                env_border + agt_radius) \
                or position[0] >= self.width - (env_border + agt_radius) or position[1] >= self.length - (
                env_border + agt_radius):
            return False

        # Check if the position falls on an obstacle
        try:
            if self.obstacles is not None:
                for obstacle in self.obstacles:
                    if obstacle['range_width'][0] - agt_radius <= position[0] <= obstacle['range_width'][1] + agt_radius \
                            and obstacle['range_length'][0] - agt_radius <= position[1] <= obstacle['range_length'][
                        1] + agt_radius:
                        # print('Position:', position, ' -------> ', 'Falling in this obstacle', obstacle)
                        return False
        except:
            return True
        return True

    def render(self, sources, agents, filename=''):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([self.origin[0], self.origin[0] + self.width])
        ax.set_ylim([self.origin[1], self.origin[1] + self.length])

        border_line_width = self.obj_size["border_width"]
        # plot environment bounders
        for bord_orig, bord_width, bord_height in zip(
                [self.origin, self.origin, [self.origin[0], self.length - border_line_width],
                 [self.width - border_line_width, self.origin[1]]],
                [self.width, border_line_width, self.width, border_line_width],
                [border_line_width, self.length, border_line_width, self.length]
        ):
            ax.add_patch(
                patches.Rectangle(bord_orig, bord_width, bord_height, alpha=1.0, edgecolor='k', facecolor='black')
            )

        # plot obstacles
        if self.obstacles is not None:
            for obs in self.obstacles:
                origin = (obs['origin'][0], obs['origin'][1])
                width = obs['range_width'][1] - obs['range_width'][0]
                height = obs['range_length'][1] - obs['range_length'][0]
                # ax.add_patch(
                #    patches.Rectangle(origin, width, height, alpha=0.5, edgecolor='k', facecolor='tab:gray', hatch='/')
                # )
                ax.add_patch(
                    patches.Rectangle(origin, width, height, alpha=0.8, edgecolor='k', facecolor='black')
                )

        # plot data sources
        for src in sources.values():
            center = (src.position[0], src.position[1])
            ax.add_patch(patches.Circle(center, radius=self.obj_size["source_radius"], alpha=0.8, facecolor='r'))
        # plot agents
        for ag in agents.values():
            center = (ag.current_position[0], ag.current_position[1])
            ax.add_patch(patches.Circle(center, radius=self.obj_size["agent_radius"], alpha=1.0, facecolor='b'))
        plt.axis('off')
        if len(filename) != 0:
            dir_name = 'rendered_steps/'  # Intended directory for output files
            save_file_name = os.path.join(dir_name, f"{filename}.png")
            plt.savefig(save_file_name)
        # plt.show()
        plt.close('all')


class AllAgents:
    def __init__(self, physical_size, type_reward='global_reward'):
        self.type_reward = type_reward
        self.physical_size = physical_size
        self.sources = {}
        self.agents = {}
        self.sources_coverage = {}
        self.sources_coverage_only_me = {}
        self.sources_coverage_without_me = {}

    def add_agent(self, agent, agent_type='source'):
        if agent_type != 'source':
            self.agents[agent.id] = agent
        else:
            self.sources[agent.id] = agent

    def compute_source_coverage__(self, a, n, lamdba_w, lambda_n, agent_id=None):
        """
        Compute v'_jt and C_jt for all the sources (equation 2 and 3 in the paper)
        When agent Id is provided, it compute the coverage of the sources only by that agent otherwise all agents
        :return:
        """
        if len(self.sources_coverage) == 0:
            self.sources_coverage = {}

        for source in self.sources.values():
            vprime_source = {}
            c_source = {}
            for info_type in source.info_strength.keys():
                vprime_source[info_type] = source.info_strength[info_type] + self.compute_agents_coverage(
                    source.position, info_type, a, n, lamdba_w, lambda_n, agent_id
                )
                c_source[info_type] = ((source.info_strength[info_type] - vprime_source[info_type]) /
                                       source.info_strength[
                                           info_type]) * 100
            self.sources_coverage[source.id] = c_source

    def compute_source_coverage(self):
        """
        Compute v'_jt and C_jt for all the sources (equation 2 and 3 in the paper)
        When agent Id is provided, it compute the coverage of the sources only by that agent otherwise all agents
        :return:
        """
        if len(self.sources_coverage) == 0:
            self.sources_coverage = {}

        for source in self.sources.values():
            vprime_source = {}
            c_source = {}
            for info_type in source.info_strength.keys():
                # vprime_source[info_type] = source.info_strength[info_type] + self.compute_agents_coverage(
                vprime_source[info_type] = self.compute_agents_coverage(
                    source, info_type
                )
            #    c_source[info_type] = ((source.info_strength[info_type] - vprime_source[info_type]) /
            #                           source.info_strength[
            #                               info_type]) * 100
            # self.sources_coverage[source.id] = c_source
            self.sources_coverage[source.id] = vprime_source

    def compute_source_coverage_only_agent_i(self, agent_id):
        """
        Compute v'_jt and C_jt for all the sources (equation 2 and 3 in the paper)
        When agent Id is provided, it compute the coverage of the sources only by that agent otherwise all agents
        :return:
        """
        for source in self.sources.values():
            vprime_source = {}
            c_source = {}
            for info_type in source.info_strength.keys():
                #vprime_source[info_type] = source.info_strength[info_type] + self.compute_agents_coverage(
                vprime_source[info_type] = self.compute_agents_coverage(
                    source, info_type, agent_id
                )
                #c_source[info_type] = ((source.info_strength[info_type] - vprime_source[info_type]) /
                #                       source.info_strength[
                #                           info_type]) * 100
            #self.sources_coverage_only_me[source.id] = c_source
            self.sources_coverage_only_me[source.id] = vprime_source

    def compute_source_coverage_without_agent_i(self, agent_id):
        """
        Compute v'_jt and C_jt for all the sources (equation 2 and 3 in the paper)
        When agent Id is provided, it compute the coverage of the sources only by that agent otherwise all agents
        :return:
        """
        if agent_id is None:
            return ValueError("Agent id for coverage with him missing")

        for source in self.sources.values():
            vprime_source = {}
            c_source = {}
            for info_type in source.info_strength.keys():
                #vprime_source[info_type] = source.info_strength[info_type] + self.compute_agents_coverage(
                vprime_source[info_type] = self.compute_agents_coverage(
                    source, info_type, agent_id
                )
                #c_source[info_type] = ((source.info_strength[info_type] - vprime_source[info_type]) /
                #                       source.info_strength[
                #                           info_type]) * 100
            # self.sources_coverage_without_me[source.id] = c_source
            self.sources_coverage_without_me[source.id] = vprime_source

    def compute_agents_coverage___(self, source_position, info_type, a, n, lamdba_w, lambda_n, agent_id=None):
        """
        Compute the summation term in Equation 2
        :param agent_id:
        :param source_position:
        :return:
        """
        sum_coverage = 0
        if self.type_reward == 'local_reward' and agent_id is not None:
            # compute the coverage for only the agent whose id is given (for local reward calculation)
            agent = self.agents[agent_id]
            agent_position = agent.current_position
            agent_i_coverage = agent.info_efficiency[info_type] * utils.dropoff_function(
                source_position, agent_position, a, n, lamdba_w, lambda_n
            )
            return agent_i_coverage
        elif self.type_reward == 'difference_reward' and agent_id is not None:
            # compute the coverage for the entire system excluding agent i (for system performance without agent i)
            for agent in self.agents.values():
                if agent.id != agent_id:
                    agent_position = agent.current_position
                    sum_coverage = sum_coverage + agent.info_efficiency[info_type] * utils.dropoff_function(
                        source_position, agent_position, a, n, lamdba_w, lambda_n
                    )
            return sum_coverage
        else:
            # compute the coverage for the entire system (for global system performance)
            for agent in self.agents.values():
                agent_position = agent.current_position
                sum_coverage = sum_coverage + agent.info_efficiency[info_type] * utils.dropoff_function(
                    source_position, agent_position, a, n, lamdba_w, lambda_n
                )
        return sum_coverage

    # Compute agents' coverage of source j for information type t
    def compute_agents_coverage(self, source, info_type, agent_id=None):
        """
        Compute the summation term in Equation 2
        :param agent_id:
        :param source_position:
        :return:
        """
        sum_coverage = 0
        if self.type_reward == 'local_reward' and agent_id is not None:
            # compute the coverage for only the agent whose id is given (for local reward calculation)
            agent = self.agents[agent_id]
            agent_position = agent.current_position
            agent_i_coverage = - agent.info_efficiency[info_type] / np.log(
                utils.likelihood(
                    source.position,
                    agent_position,
                    amplitude=1,
                    sigma=source.info_strength[info_type]
                )
            )
            return agent_i_coverage
        elif self.type_reward == 'difference_reward' and agent_id is not None:
            # compute the coverage for the entire system excluding agent i (for system performance without agent i)
            for agent in self.agents.values():
                if agent.id != agent_id:
                    agent_position = agent.current_position
                    sum_coverage = sum_coverage - agent.info_efficiency[info_type] / np.log(
                        utils.likelihood(
                            source.position,
                            agent_position,
                            amplitude=1,
                            sigma=source.info_strength[info_type]
                        )
                    )
            return sum_coverage
        else:
            # compute the coverage for the entire system (for global system performance)
            for agent in self.agents.values():
                agent_position = agent.current_position
                sum_coverage = sum_coverage - agent.info_efficiency[info_type] / np.log(
                    utils.likelihood(
                        source.position,
                        agent_position,
                        amplitude=1,
                        sigma=source.info_strength[info_type]
                    )
                )
        return sum_coverage

    def global_fitness(self):
        performance = 0
        normalizer = 0
        for source in self.sources.values():
            n_info_type_source = len(self.sources_coverage[source.id].values())
            performance = performance + sum(self.sources_coverage[source.id].values())
            normalizer = normalizer + n_info_type_source + 1
        return performance / normalizer

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
            # sensing all agents in each quadrant
            quadrants_agents = np.zeros((1, 4)).ravel()
            # sensing all sources in each quadrant
            quadrants_sources = np.zeros((1, 4)).ravel()
            max_distance_to_other_agents = np.zeros((1, 4)).ravel() + 0.01 # added to avoid division by zero
            for agent in self.agents.values():
                if agent.id != target_agent.id:
                    distance_aj = utils.euclidian_distance(agent.current_position, target_agent.current_position) + 1
                    # Quadrant 0
                    if agent.current_position[0] < target_agent.current_position[0] and \
                            agent.current_position[1] < target_agent.current_position[1]:
                        quadrants_agents[0] = quadrants_agents[0] + distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if distance_aj > max_distance_to_other_agents[0]:
                            max_distance_to_other_agents[0] = distance_aj

                    # Quadrant 1
                    elif agent.current_position[0] >= target_agent.current_position[0] and \
                            agent.current_position[1] < target_agent.current_position[1]:
                        quadrants_agents[1] = quadrants_agents[1] + distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if distance_aj > max_distance_to_other_agents[1]:
                            max_distance_to_other_agents[1] = distance_aj

                    # Quadrant 2
                    elif agent.current_position[0] >= target_agent.current_position[0] and \
                            agent.current_position[1] >= target_agent.current_position[1]:
                        quadrants_agents[2] = quadrants_agents[2] + distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if distance_aj > max_distance_to_other_agents[2]:
                            max_distance_to_other_agents[2] = distance_aj

                    # Quadrant 3
                    else:
                        quadrants_agents[3] = quadrants_agents[3] + distance_aj

                        # Check if the value of the information type for agent i is greater than max
                        if distance_aj > max_distance_to_other_agents[3]:
                            max_distance_to_other_agents[3] = distance_aj
            sensing_agents_quadrants[info_type] = np.divide(quadrants_agents, max_distance_to_other_agents)

            max_distance_to_sources = np.zeros((1, 4)).ravel() + 0.01 # added to avoid division by zero
            target_agent_info_efficiency = target_agent.info_efficiency[info_type]
            for source in self.sources.values():
                distance_ij = utils.euclidian_distance(source.position, target_agent.current_position) + 1

                neglog_likelihood = - np.log(
                    utils.likelihood(
                        source.position,
                        target_agent.current_position,
                        amplitude=1,
                        sigma=source.info_strength[info_type]
                    )
                )

                # Quadrant 0
                if source.position[0] < target_agent.current_position[0] and \
                        source.position[1] < target_agent.current_position[1]:
                    #current_src_update = (target_agent_info_efficiency * distance_ij) / neglog_likelihood
                    current_src_update = target_agent_info_efficiency / neglog_likelihood
                    quadrants_sources[0] = quadrants_sources[0] + current_src_update

                    ## Check if the value of the information type for agent i is greater than max
                    #if distance_ij > max_distance_to_sources[0]:
                    #    max_distance_to_sources[0] = distance_ij

                # Quadrant 1
                elif source.position[0] >= target_agent.current_position[0] and \
                        source.position[1] < target_agent.current_position[1]:
                    #current_src_update = (target_agent_info_efficiency * distance_ij) / neglog_likelihood
                    current_src_update = target_agent_info_efficiency / neglog_likelihood
                    quadrants_sources[1] = quadrants_sources[1] + current_src_update

                    ## Check if the value of the information type for agent i is greater than max
                    #if distance_ij > max_distance_to_sources[1]:
                    #    max_distance_to_sources[1] = distance_ij

                # Quadrant 2
                elif source.position[0] >= target_agent.current_position[0] and \
                        source.position[1] >= target_agent.current_position[1]:
                    #current_src_update = (target_agent_info_efficiency * distance_ij) / neglog_likelihood
                    current_src_update = target_agent_info_efficiency / neglog_likelihood
                    quadrants_sources[2] = quadrants_sources[2] + current_src_update

                    ## Check if the value of the information type for agent i is greater than max
                    #if distance_ij > max_distance_to_sources[2]:
                    #    max_distance_to_sources[2] = distance_ij

                # Quadrant 3
                else:
                    #current_src_update = (target_agent_info_efficiency * distance_ij) / neglog_likelihood
                    current_src_update = target_agent_info_efficiency / neglog_likelihood
                    quadrants_sources[3] = quadrants_sources[3] + current_src_update

                    ## Check if the value of the information type for agent i is greater than max
                    #if distance_ij > max_distance_to_sources[3]:
                    #    max_distance_to_sources[3] = distance_ij

            #sensing_sources_quadrants[info_type] = np.divide(quadrants_sources, max_distance_to_sources)
            sensing_sources_quadrants[info_type] = quadrants_sources

        return sensing_agents_quadrants, sensing_sources_quadrants

    def check_position_sources(self, position, target_agent_id):
        """
        Check if the given position overlap with a source or other agents
        :param check_obstacle:
        :param position:
        :return:
        """
        target_agent = self.agents[target_agent_id]
        agt_radius = self.physical_size['agent_radius']
        src_radius = self.physical_size['source_radius']
        margin = src_radius
        for agent in self.agents.values():
            if agent.id != target_agent.id:
                # Check if the position falls on a another agent
                distance_target_to_agent_i = utils.euclidian_distance(
                    agent.current_position,
                    position
                )
                if distance_target_to_agent_i < 2 * agt_radius + margin:
                    return False
        for source in self.sources.values():
            # Check if the position falls on a source
            distance_target_to_source_j = utils.euclidian_distance(
                source.position,
                position
            )
            if distance_target_to_source_j < agt_radius + src_radius + margin:
                return False

        return True


    '''
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
            max_info_sensing_agents = np.ones((1, 4)).ravel() * target_agent.info_efficiency[info_type]
            max_info_sensing_sources = np.ones((1, 4)).ravel() * target_agent.info_efficiency[info_type]
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
    '''
