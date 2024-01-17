
# shelang line is necsssary to run the scripts using subprocess so we do not need to activate conda environments
import sys
import os

sys.path.append(os.path.dirname(__file__))
print(os.path.dirname(__file__))

import numpy as np
from model.argoverse_baseline.constant_velocity import ConstantVelocityPlanner
from model.argoverse_baseline.constant_acceleration import ConstantAccelerationPlanner
from simulator.knearestneighbor.knn_planner import KNNPlanner
#from simulator.LSTM.lstm_simulator import LSTM
#from simulator.LaneGCN.lanegcn_simulator import LaneGCN
#from simulator.HiVT.hivt_simulator import HiVT

import logging
import time
import subprocess
import warnings
warnings.filterwarnings('ignore')

# This must be same as xxxaaa.py
MAX_HISTORY_MOTION_PREDICTION = 20 # This value is same in moped_param.h


#  The planner wrapper returns numpy array shape [number_agents, pred_len, 2] with correponding probability for each agent


class PlannerWrapper():
    def __init__(self, pred_len = 30):
        self.pred_len = pred_len
        #self.cv = ConstantVelocityPlanner() #Run okay with pred_len
        #self.ca = ConstantAccelerationPlanner() #Run okay with pred_len
        #self.laneGCN = LaneGCN(self.get_most_free_gpu_device()) # Always return 30, no need to do anything
        #self.hivt = HiVT() # Alwasy return 30, no need to do anything
        # self.knn_default = KNNPlanner(use_social=False)
        self.knn_social = KNNPlanner(use_social=True)
        #self.lstm_default = LSTM(use_social=False)
        #self.lstm_social = LSTM(use_social=True)

        self.model_running = self.knn_social.__class__.__name__

    def get_most_free_gpu_device(self):
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        used_gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        used_gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in used_gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        total_gpu_info = list(filter(lambda info: "Total" in info, gpu_info))
        total_gpu_info =  [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in total_gpu_info
        ]

        free_gpus = [
            (total-used, i) for i, (total, used) in enumerate(zip(total_gpu_info, used_gpu_info))
        ]
        print("Total GPU info: ", free_gpus)

        best_memory, best_device = max(free_gpus)
        
        return best_device

    def constant_velocity(self, obs_trajectory: np.ndarray):
        '''
        obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
        return [num_trajectory, pred_len, 2]
        '''
        # predictions must have shape [num_trajectory, pred_len, 2]

        predictions = self.cv.predict(obs_trajectory, pred_len=self.pred_len, avg_point_list=(20,))
        #print(predictions)
        # Constant velocity gives a deterministic probabilities
        probs = np.ones(shape=predictions.shape[0])


        return probs, predictions

    def constant_acceleration(self, obs_trajectory):
        '''
        obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
        '''
        predictions = self.ca.predict(obs_trajectory, pred_len=self.pred_len, avg_point_list=(20,))

        predictions = predictions[0]
        # After this, prediction should have shape [num_trajectory, 1, 2], where 1 is first prediction
        # Constant velocity gives a deterministic probabilities
        probs = np.ones(shape=predictions.shape[0])

        return probs, predictions

    def lanegcn_prediction(self, obs_trajectory):
        preds = self.laneGCN.run(obs_trajectory)
        probs = np.array([1/ preds.shape[1]] * preds.shape[0], dtype=float)
        return probs, preds[:, 0, :, :] # sliding over agents, choose K=0, all prediction steps, all x and y

    def hivt_prediction(self, obs_trajectory):
        preds = self.hivt.run(obs_trajectory)
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)
        return probs, preds[:, 0, :, :]  # sliding over agents, choose K=0, all prediction steps, all x and y

    def lstm_default_prediction(self, obs_trajectory):
        # Shape [num_trajectory, 1, 2]
        preds = self.lstm_default.run(obs_trajectory, pred_len=self.pred_len)
        # Shape [num_trajectory]
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)

        return probs, preds  # sliding over agents, choose K=0, all prediction steps, all x and y


    def lstm_social_prediction(self, obs_trajectory):
        # Shape [num_trajectory, 1, 2]
        preds = self.lstm_social.run(obs_trajectory, pred_len=self.pred_len)
        # Shape [num_trajectory]
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)
        return probs, preds  # sliding over agents, choose K=0, all prediction steps, all x and y

    def knn_default_prediction(self, obs_trajectory):
        # Shape [num_trajectory, 1, 2]
        preds = self.knn_default.predict_default(obs_trajectory, 
                                        pred_len=self.pred_len, obs_len=MAX_HISTORY_MOTION_PREDICTION)
        # Shape [num_trajectory]
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)
        return probs, preds  # sliding over agents, choose K=0, all prediction steps, all x and y


    def knn_social_prediction(self, obs_trajectory):
        # Shape [num_trajectory, 1, 2]
        preds = self.knn_social.predict_default(obs_trajectory, 
                                        pred_len=self.pred_len, obs_len=MAX_HISTORY_MOTION_PREDICTION)
        # Shape [num_trajectory]
        probs = np.array([1 / preds.shape[1]] * preds.shape[0], dtype=float)

        return probs, preds  # sliding over agents, choose K=0, all prediction steps, all x and y

    def do_predictions(self, obs_trajectory):
        '''
                obs_trajectory should have shape [num_trajectory, obs_history, 2] (where num_trajectory = number_agents)
                return [num_trajectory, pred_len, 2]
                return:
                    probs (num_trajectory,)
                    preds (num_trajectory, n, 2), where n is number of prediction steps
        '''
        #assert len(obs_trajectory.shape) == 3 and obs_trajectory.shape[1] == MAX_HISTORY_MOTION_PREDICTION and obs_trajectory.shape[2] == 2

        #probs, preds = self.constant_velocity(obs_trajectory)
        #probs, preds = self.constant_acceleration(obs_trajectory)
        #probs, preds = self.lstm_default_prediction(obs_trajectory)
        #probs, preds = self.lstm_social_prediction(obs_trajectory)
        #probs, preds = self.knn_default_prediction(obs_trajectory)
        probs, preds = self.knn_social_prediction(obs_trajectory)
        #probs, preds = self.hivt_prediction(obs_trajectory)
        #probs, preds = self.lanegcn_prediction(obs_trajectory)

        #assert len(probs.shape) == 1
        #assert len(preds.shape) == 3 and preds.shape[0] == obs_trajectory.shape[0] and preds.shape[1] >= 1 and preds.shape[2] == 2

        return probs, preds

if __name__ == "__main__":
    realistic_sim_bounds = {'beijing': [770.7, -4.7952, 3234.3, 211.5],
                            'chandni_chowk': [271.5, 786.94, 535.68, 1180.6],
                            'magic': [31.333, 77.263, 312.23, 365.09],
                            'shi_men_er_lu': [818.3, 1699.1, 1187.0, 2019.3]}

    import csv
    import numpy as np
    filename = 'examples/pomdp_search_log-0_0_pid-56_r_haha-150.csv'
    n_obs = 20
    # Initialize the dictionary
    agent_dict = {}
    gt_dict = {}
    start = {}
    end = {}
    start_gt = {}
    end_gt = {}
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        # Iterate over each row in the CSV file
        for row in reader:
            # Get the track ID and x,y coordinates from the row
            track_id = int(row['TRACK_ID'])
            x = float(row['X'])
            y = float(row['Y'])
            timestamp = int(row['TIMESTAMP'])
            if timestamp > 19:
                if track_id in gt_dict:
                    gt_dict[track_id][timestamp-20][0] = x
                    gt_dict[track_id][timestamp-20][1] = y
                else:
                    gt_dict[track_id] = np.zeros([30,2])
                    gt_dict[track_id][timestamp-20][0] = x
                    gt_dict[track_id][timestamp-20][1] = y
            else:
                if track_id in agent_dict:
                    agent_dict[track_id][timestamp][0] = x
                    agent_dict[track_id][timestamp][1] = y
                    end[track_id] = timestamp
                else:
                    agent_dict[track_id] = np.zeros([20,2])
                    agent_dict[track_id][timestamp][0] = x
                    agent_dict[track_id][timestamp][1] = y
                    start[track_id] = timestamp
    # Convert the dictionary values to numpy arrays
    for track_id, obs_list in agent_dict.items():
        obs_list[:start[track_id]] = obs_list[start[track_id]]
        obs_list[end[track_id]:] = obs_list[end[track_id]]

    trajectories = np.stack(list(agent_dict.values()))
    gts = np.stack(list(gt_dict.values()))
    
    planner = PlannerWrapper()

    probs, preds = planner.do_predictions(trajectories)
    
    for i in range(20):
        mask = gts[i][:,0] != 0
        ADE = np.sqrt(((preds[i][mask]-gts[i][mask])**2).sum(1))
        FDE = ADE[-1]
        ADE = np.mean(ADE)
        # print(trajectories[i])
        # print(preds[i])
        # print(gts[i])
        print(f"prediction length {len(gts[i])}")
        print(f"ADE for the first agent is {ADE}")
        print(f"FDE for the first agent is {FDE}")