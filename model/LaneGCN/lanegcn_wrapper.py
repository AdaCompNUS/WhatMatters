import shutil
import tempfile
import zipfile

import numpy as np
from typing import List, Tuple, Dict
from argoverse.evaluation.competition_util import generate_forecasting_h5
import os
import sys

from utils.baseline_config import FEATURE_FORMAT
import pandas as pd
import pickle as pkl

import yaml
from utils.logger import Logger
import numpy as np

from model.base_model import MotionPrediction


import subprocess



class LaneGCN(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.dataset_type = self.config.data.dataset_type
        self.train_folder = self.config.data.train.raw_dir
        self.val_folder = self.config.data.val.raw_dir

        self.val_features_path = self.config.data.val.features
        self.train_features_path = self.config.data.train.features

        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len
        self.gpus = self.config.model.gpus

        self.train_cfg  = self.config.model.train_cfg
        self.val_cfg = self.config.model.val_cfg

        if not os.path.exists(self.train_features_path):
            print("Generating Train Features for LaneGCN")
            subprocess.call(['python', 'model/LaneGCN/preprocess_data.py', '--raw_folder', self.train_folder, 
            '--features_path', self.train_features_path, '--dataset_type', self.dataset_type, '--mode', 'train'])
        
        if not os.path.exists(self.val_features_path):
            print("Generating Val Features for LaneGCN")
            subprocess.call(['python', 'model/LaneGCN/preprocess_data.py', '--raw_folder', self.val_folder, 
            '--features_path', self.val_features_path, '--dataset_type', self.dataset_type, '--mode', 'val'])

    def train(self):
        subprocess.run(['horovodrun', '-np', str(self.gpus), 'python', 'model/LaneGCN/train_lanegcn.py', 
        '--train_dir', self.train_folder, '--val_dir', self.val_folder, '--split', 'train', '--obs_len', str(self.obs_len), 
        '--pred_len', str(self.pred_len), '--raw_train_dir', self.train_features_path, '--raw_val_dir', self.val_features_path, 
        '--weight', self.train_cfg.weight, '--resume', self.train_cfg.resume])

    def validate(self):
        subprocess.call(['horovodrun', '-np', str(self.gpus), 'python', 'model/LaneGCN/train_lanegcn.py',  
        '--train_dir', self.train_folder, '--val_dir', self.val_folder, '--split', 'val', '--obs_len', str(self.obs_len), 
        '--pred_len', str(self.pred_len), '--raw_train_dir', self.train_features_path, '--raw_val_dir', self.val_features_path, 
        '--weight', self.val_cfg.weight, '--resume', self.val_cfg.resume])
    
    def test(self):
        pass

