import numpy as np


class AffinityCalculator:
    def __init__(self, config, config_label):
        self.funcsets = {"sta_method": self.sta_method}
        self.affinity_scoring_method = config.get(config_label, "affinity_scoring_method")
        self.config = config
        self.config_label = config_label

    def calculate(self, bbox1, bbox2):
        return self.funcsets[self.affinity_scoring_method](bbox1, bbox2)

    def sta_method(self, bbox1, bbox2):
        """Use appearance, temporal and spatial feature to calculating Distance between 2 bboxes"""
        spatial_decay_rate = float(self.config.get(self.config_label, "spatial_decay_rate"))
        time_decay_rate = float(self.config.get(self.config_label, "time_decay_rate"))
        d1 = (bbox1.left + bbox1.width / 2) - (bbox2.left + bbox2.width / 2)
        d2 = (bbox1.top + bbox1.height / 2) - (bbox2.top + bbox2.height / 2)
        d = np.sqrt(d1 * d1 + d2 * d2)
        d = d / np.sqrt((bbox1.area + bbox2.area)/2)
        spatial_factor = max(0, (2 - np.exp(d / spatial_decay_rate)))
        width_ratio = float(min(bbox1.width, bbox2.width)) / max(bbox1.width, bbox2.width)
        height_ratio = float(min(bbox1.height, bbox2.height)) / max(bbox1.height, bbox2.height)
        size_factor = width_ratio * height_ratio

        temporal_factor = max(0, (2 - np.exp((bbox1.frame_num - bbox2.frame_num) / time_decay_rate)))
        hist_minimum = np.minimum(bbox1.feature, bbox2.feature)
        hist_maximum = np.maximum(bbox1.feature, bbox2.feature)
        appearance_hist_rate = np.zeros(len(bbox1.feature))
        for i in range(len(bbox1.feature)):
            if not hist_maximum[i]:
                appearance_hist_rate[i] = 1
            else:
                appearance_hist_rate[i] = hist_minimum[i] / hist_maximum[i]
        appearance_factor = np.average(appearance_hist_rate)
        affinity = appearance_factor * spatial_factor * size_factor * temporal_factor
        return affinity


