#!/usr/bin/env python3
# @author:

import numpy as np

class Source:
    def __init__(self, source_id, position, info_strength=None):
        if info_strength is None:
            info_strength = {'precipitation': 20, 'temperature': 15}
        self.id = source_id
        self.intialt_position = position
        self.position = position  # 2d vector containing x0 and y0
        # if np.isscalar(info_strength):
        self.info_strength = info_strength  # dictionary -> {'pr': 20, 'tair': 15}

    def move(self, env, all_agents_only):
        new_x = self.position[0] + np.random.uniform(-5, 5)
        new_y = self.position[1] + np.random.uniform(-5, 5)

        if not env.check_position_source([new_x, new_y], all_agents_only):
            self.move(env, all_agents_only)
        else:
            self.position[0] = new_x
            self.position[1] = new_y

    def reset_source(self):
        self.position = self.intialt_position
