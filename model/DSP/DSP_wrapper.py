import shutil
import tempfile
import zipfile

import numpy as np
from typing import List, Tuple, Dict
from argoverse.evaluation.competition_util import generate_forecasting_h5
import os

from utils.baseline_config import FEATURE_FORMAT
import pandas as pd
import pickle as pkl
print(os.getcwd())
import yaml
from utils.logger import Logger
import numpy as np

from model.base_model import MotionPrediction

from model.DSP.evaluation import main as main_eval
from model.DSP.visualize import main as main_test
from model.DSP.train import main as main_train

class DSP(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.feature_dir = self.config.data.feature_dir
        self.adv_cfg_path = self.config.data.adv_cfg_path
        self.dataset = self.config.data.dataset_type

        self.model = self.config.model.name
        self.loss = self.config.model.loss
        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len
        self.use_cuda = self.config.model.use_cuda
        self.model_path = self.config.model.model_path

        self.train_cfg  = self.config.model.train_cfg
        self.val_cfg = self.config.model.val_cfg.val_batch_size
        self.test_cfg = self.config.model.test_cfg.shuffle

    def train(self):
        main_train(feature_dir = self.feature_dir, adv_cfg_path = self.adv_cfg_path,
                model = self.model, loss = self.loss,
                obs_len = self.obs_len, pred_len = self.pred_len,
                use_cuda = self.use_cuda, model_path = self.model_path, train_cfg = self.train_cfg, dataset = self.dataset)


    def test(self):
        main_test(feature_dir = self.feature_dir, adv_cfg_path = self.adv_cfg_path,
                model = self.model, loss = self.loss,
                obs_len = self.obs_len, pred_len = self.pred_len,
                use_cuda = self.use_cuda, model_path = self.model_path, shuffle = self.test_cfg)


    def validate(self):

        main_eval(feature_dir = self.feature_dir, adv_cfg_path = self.adv_cfg_path,
                model = self.model, loss = self.loss,
                obs_len = self.obs_len, pred_len = self.pred_len,
                use_cuda = self.use_cuda, model_path = self.model_path, val_batch_size = self.val_cfg)

