
# shelang line is necsssary to run the scripts using subprocess so we do not need to activate conda environments
import sys
import os

#sys.path.append(os.path.dirname(__file__))
#print(os.path.dirname(__file__))

import numpy as np
from model.argoverse_baseline.constant_velocity import ConstantVelocityPlanner
from model.argoverse_baseline.constant_acceleration import ConstantAccelerationPlanner
from model.argoverse_baseline.summit_knn import KNNPlanner
from simulator.LaneGCN.lanegcn_simulator import LaneGCN
from simulator.HiVT.hivt_simulator import HiVT

import warnings
warnings.filterwarnings('ignore')

import logging
import time

# This must be same as xxxaaa.py
MAX_HISTORY_MOTION_PREDICTION = 20 # This value is same in moped_param.h


#  The planner wrapper returns numpy array shape [number_agents, pred_len, 2] with correponding probability for each agent


class PlannerWrapper():
    def __init__(self, pred_len = 1):
        self.pred_len = pred_len
        #self.laneGCN = LaneGCN()
        self.hivt = HiVT()

    def constant_velocity(self, obs_trajectory: np.ndarray):
        '''
        obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
        return [num_trajectory, pred_len, 2]
        '''
        # predictions must have shape [num_trajectory, pred_len, 2]
        predictions = ConstantVelocityPlanner.predict(obs_trajectory, pred_len=self.pred_len, avg_point_list=(20,))
        #print(predictions)
        # Constant velocity gives a deterministic probabilities
        probs = np.ones(shape=predictions.shape[0])

        return probs, predictions

    def constant_acceleration(self, obs_trajectory):
        '''
        obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
        '''
        predictions = ConstantAccelerationPlanner.predict(obs_trajectory, pred_len=self.pred_len, avg_point_list=(20,))
        #print(predictions)

        predictions = predictions[0]
        # After this, prediction should have shape [num_trajectory, 1, 2], where 1 is first prediction
        # Constant velocity gives a deterministic probabilities
        probs = np.ones(shape=predictions.shape[0])

        return probs, predictions

    def knn_map_nosocial(self, obs_trajectory):
        predictions = KNNPlanner.predict_map_nosocial(obs_trajectory, pred_len=30)
        #predictions = predictions[:, 0, :] # Get all agents, just first time step, at all x,y axis
        probs = np.ones(shape=predictions.shape[0])

        return probs, predictions

    def lanegcn_prediction(self, obs_trajectory):
        preds = self.laneGCN.run(obs_trajectory)
        probs = np.array([1/ preds.shape[1]] * preds.shape[0], dtype=float)
        #print(probs)
        return probs, preds[:, 0, :, :] # sliding over agents, choose K=0, all prediction steps, all x and y

    def hivt_prediction(self, obs_trajectory):
        preds = self.hivt.run(obs_trajectory)
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)
        print(preds.shape)
        return probs, preds[:, 0, :, :]  # sliding over agents, choose K=0, all prediction steps, all x and y

    def do_predictions(self, obs_trajectory):
        '''
                obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
                return [num_trajectory, pred_len, 2]
                return:
                    probs (num_trajectory,)
                    preds (num_trajectory, n, 2), where n is number of prediction steps
        '''
        assert len(obs_trajectory.shape) == 3 and obs_trajectory.shape[1] == MAX_HISTORY_MOTION_PREDICTION and obs_trajectory.shape[2] == 2
        #logging.info(obs_trajectory)

        #probs, preds = self.constant_velocity(obs_trajectory)

        #probs, preds =  self.constant_acceleration(obs_trajectory)
        #return self.knn_map_nosocial(obs_trajectory)
        #probs, preds = self.lanegcn_prediction(obs_trajectory)
        probs, preds = self.hivt_prediction(obs_trajectory)

        assert len(probs.shape) == 1
        assert len(preds.shape) == 3 and preds.shape[0] == obs_trajectory.shape[0] and preds.shape[1] >= 1 and \
               preds.shape[2] == 2
        return probs, preds

if __name__ == "__main__":
    realistic_sim_bounds = {'beijing': [770.7, -4.7952, 3234.3, 211.5],
                            'chandni_chowk': [271.5, 786.94, 535.68, 1180.6],
                            'magic': [31.333, 77.263, 312.23, 365.09],
                            'shi_men_er_lu': [818.3, 1699.1, 1187.0, 2019.3]}

    trajectories = np.random.normal(loc=[150,200], scale=1, size=(20,20,2))


    #temp = PlannerWrapper().knn_map_nosocial(trajectories)
    begin1 = time.time()
    probs, preds = PlannerWrapper().do_predictions(trajectories)
    begin2 = time.time()
    # print(f"Prediction time for Constant Velocity for 1 preds length: {begin2 - begin1}")

    # temp = PlannerWrapper().do_predictions(trajectories)
    # print(f"Prediction time for Constant Velocity for 1 preds length: {time.time() - begin2}")
    #print(preds)
    #print(preds.shape)
    #print(probs.shape)

    #print(temp)