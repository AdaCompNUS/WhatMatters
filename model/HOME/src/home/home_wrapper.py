import shutil
import tempfile
import zipfile

import numpy as np
from typing import List, Tuple, Dict
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



class HOME(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.train_features_path = self.config.data.train.features
        self.val_features_path = self.config.data.val.features

        self.config_path = self.config.model.config_path
        self.preprocess_path = self.config.model.prepath
        self.train_heatmap_path = self.config.model.heatmappath
        self.predictor_path = self.config.model.predictorpath
        self.val_path = self.config.model.valpath

        if not os.path.exists(self.train_features_path):
            print("Generating Train Features for HOME")
            subprocess.call(['python3', self.preprocess_path, '--cfg', self.config_path])
        
        if not os.path.exists(self.val_features_path):
            print("Generating Val Features for HOME")
            subprocess.call(['python3', self.preprocess_path, '--cfg', self.config_path])

    def train(self):
        print("Training Heatmap")
        subprocess.call(['python3', self.train_heatmap_path, '--cfg', self.config_path])
        print("Training Predictor")
        subprocess.call(['python3', self.predictor_path, '--cfg', self.config_path])
        
    def validate(self):
        subprocess.call(['python3', self.val_path, '--cfg', self.config_path])
    
    def test(self):
        pass

