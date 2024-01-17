import argparse
#from model.argoverse_baseline.knearest_neighbor import KNearestNeighbor
#from model.argoverse_baseline.constant_velocity import ConstantVelocity
#from model.LSTM.lstm_wrapper import LSTM
# from model.gamma.gamma_wrapper import Gamma
# from model.LaneGCN.lanegcn_wrapper import LaneGCN
# from model.DSP.DSP_wrapper import DSP
import os
import shutil
from typing import Any
from mmcv import Config
from utils.summit_dataloader import SummitBaselineDataset

def parse_args():
    parser = argparse.ArgumentParser(description="running a motion prediction model in Summit simulator")

    parser.add_argument('config', help='training config file')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')


    args = parser.parse_args()

    return args

def construct_model(cfg):
    # To avoid python version conflicts library, we only import the model based on the name
    model_name = cfg.model.type
    model = None
    if model_name == "LaneGCN":
        from simulator.LaneGCN.lanegcn_simulator import LaneGCN
        model = LaneGCN(config=cfg)
    if model_name == "HiVT":
        from simulator.HiVT.hivt_simulator import HiVT
        model = HiVT()
    if model_name == "LSTM":
        from simulator.LSTM.lstm_simulator import LSTM
        model = LSTM()
    return model

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    model = construct_model(cfg)
    model.run()

if __name__ == "__main__":
    main()
