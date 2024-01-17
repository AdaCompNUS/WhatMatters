from model.base_model import MotionPrediction
import pickle
import pandas as pd
import argparse
from mmcv import Config

class OnlyEvaluation(MotionPrediction):
    def __init__(self, config):
        self.config = config
        self.val_features_path = config.data.val.features

        self.obs_len = config.model.obs_len
        self.pred_len = config.model.pred_len

    def evaluate(self, forecasted_trajectories_path: str, forecasted_probabilities_path: str):
        with open(forecasted_trajectories_path, "rb") as f:
            forecasted_trajectories = pickle.load(f)

        with open(forecasted_probabilities_path, "rb") as f:
            forecasted_probabilities = pickle.load(f)

        groundtruth_trajectories = self.get_gt_trajectories(self.val_features_path,
                                                            obs_len=self.obs_len, pred_len=self.pred_len)
        data_features_frame = pd.read_pickle(self.val_features_path)

        self.evaluation_trajectories(groundtruth_trajectories=groundtruth_trajectories,
                                     forecasted_trajectories=forecasted_trajectories,
                                     features_df=data_features_frame, args=self.config,
                                     output_path="/home/cunjun/p3_catkin_ws_new/src/moped/moped_implementation/work_dirs/gamma_default/",
                                     forecasted_probabilities=forecasted_probabilities)

    def train(self):
        pass

    def test(self):
        pass

    def validate(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating testing results of a motion prediction model")

    parser.add_argument('config', help='validation config file')
    parser.add_argument('--work-dir',
                        help='the directory to save the file containing evaluation metrics')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    eval = OnlyEvaluation(cfg)
    eval.evaluate(forecasted_trajectories_path='/home/cunjun/p3_catkin_ws_new/src/moped/moped_implementation/work_dirs/gamma_default/val/prediction_trajectories.pkl',
                  forecasted_probabilities_path='/home/cunjun/p3_catkin_ws_new/src/moped/moped_implementation/work_dirs/gamma_default/val/prediction_probs.pkl')
