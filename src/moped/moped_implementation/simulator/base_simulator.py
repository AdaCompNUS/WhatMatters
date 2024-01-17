#!/home/cunjun/anaconda3/envs/conda38/bin/python
# A shebang line to run different python version

from abc import ABC, abstractmethod
import numpy as np
import logging

class Simulator(ABC):

    @abstractmethod
    def run(self, trajectories: np.ndarray):
        pass
    
    @abstractmethod
    def array2dict(self, trajectories: np.ndarray):
        pass

    @abstractmethod
    def dict2run(self, data_dict: dict):
        pass

    def preprocess(self, trajectories: np.ndarray):
        
        data_dict = self.array2dict(trajectories)
        data_run = self.dict2run(data_dict)

        return data_run

    def getCity(self, trajectories : np.ndarray):
        
        realistic_sim_bounds = {'beijing': [730.7, -20.7952, 3334.3, 250.5],
                                'chandni_chowk': [250.5, 780.94, 550.68, 1200.6],
                                'magic': [25.333, 0.0, 320.23, 370.09],
                                'shi_men_er_lu': [800.3, 1680.1, 1200.0, 2030.3]}
        mean_x, mean_y = np.mean(trajectories, axis=(0,1))
        city = None
        for k, v in realistic_sim_bounds.items():
            mean_x = float(mean_x)
            mean_y = float(mean_y)
            if v[0] <= mean_x and mean_x <= v[2] and v[1] <= mean_y and mean_y <= v[3]:
                city = k
                break
        if city is None:
            logging.info(f"Cannot found city with x {mean_x} and y {mean_y}")
        #city = 'magic'
        return city



