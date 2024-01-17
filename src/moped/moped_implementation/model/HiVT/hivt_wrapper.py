import shutil
import tempfile
import zipfile

import numpy as np
from typing import List, Tuple, Dict

from utils.baseline_config import FEATURE_FORMAT
import pandas as pd
import pickle as pkl

from utils.logger import Logger

from model.base_model import MotionPrediction

import subprocess




class HiVT(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.dataset_type = self.config.data.dataset_type
        self.root_dir = self.config.data.root_dir

        self.test_features_path = self.config.data.test.features
        self.val_features_path = self.config.data.val.features
        self.train_features_path = self.config.data.train.features

        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len

        self.train_cfg  = self.config.model.train_cfg
        self.val_cfg = self.config.model.val_cfg
        self.test_cfg = self.config.model.test_cfg 
        


    def train(self):
        subprocess.call(['python', 'model/HiVT/train.py',  '--root', self.root_dir, '--embed_dim', str(self.train_cfg.embed_dim), 
        '--train_batch_size', str(self.train_cfg.train_batch_size), '--gpus', str(self.train_cfg.gpus), '--dataset', self.dataset_type])

    def test(self):
        pass

    def validate(self):
        subprocess.call(['python', 'model/HiVT/eval.py',  '--root', self.root_dir, 
        '--batch_size', str(self.val_cfg.val_batch_size), '--ckpt_path', self.val_cfg.ckpt_path, '--dataset', self.dataset_type])


