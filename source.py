#!/usr/bin/env python3
# @author:

import numpy as np
from utils import dropoff_function, get_angle_phi


class Source:
    def __init__(self, source_id, position, info_strength=None):
        if info_strength is None:
            info_strength = {'precipitation': 20, 'temperature': 15}
        self.id = source_id
        self.position = position  # 2d vector containing x0 and y0
        # if np.isscalar(info_strength):
        self.info_strength = info_strength  # dictionary -> {'pr': 20, 'tair': 15}
