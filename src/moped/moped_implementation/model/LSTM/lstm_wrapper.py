import pickle as pkl

from typing import List, Tuple, Dict
from utils.logger import Logger
from model.base_model import MotionPrediction
from model.LSTM.lstm_train_test_gaussian import main as normal_lstm
from model.LSTM.lstm_train_test_map  import main as map_lstm

from utils.eval_forecasting_helper import evaluation


class LSTM(MotionPrediction):
    def __init__(self, config: Dict, logger: Logger):
        self.config = config
        self.logger = logger

        self.train_features_path = self.config.data.train.features
        self.val_features_path = self.config.data.val.features
        self.test_features_path = self.config.data.test.features
        
        self.lr = self.config.model.lr
        self.dataset = self.config.data.dataset_type
        self.obs_len = self.config.model.obs_len
        self.pred_len = self.config.model.pred_len
        self.normalize = self.config.model.normalize
        self.use_delta = self.config.model.use_delta
        self.use_map = self.config.model.use_map
        self.use_social = self.config.model.use_social
        

        self.train_cfg  = self.config.model.train_cfg
        self.val_cfg = self.config.model.val_cfg

    def train(self):
        if self.use_map == True:
            map_lstm(train_features = self.train_features_path, val_features = self.val_features_path, 
                obs_len = self.obs_len, pred_len = self.pred_len, normalize = self.normalize, 
                use_delta = self.use_delta, use_map = self.use_map, use_social = self.use_social, 
                lr = self.lr, cfg = self.train_cfg, test = False, dataset = self.dataset)
        else:
            normal_lstm(train_features = self.train_features_path, val_features = self.val_features_path, 
                obs_len = self.obs_len, pred_len = self.pred_len, normalize = self.normalize, 
                use_delta = self.use_delta, use_map = self.use_map, use_social = self.use_social, 
                lr = self.lr, cfg = self.train_cfg, test = False, dataset = self.dataset)

    def test(self):
        pass

    def validate(self):
        # Actually, For map-prior LSTM, we need candidate centerline to test, so we use test features here
        # test feature means val-testmode
        if self.use_map == True:
            map_lstm(train_features = self.train_features_path, val_features = self.test_features_path, 
                obs_len = self.obs_len, pred_len = self.pred_len, normalize = self.normalize, 
                use_delta = self.use_delta, use_map = self.use_map, use_social = self.use_social, 
                lr = self.lr, cfg = self.val_cfg, test = True, dataset = self.dataset)
        else:
            normal_lstm(train_features = self.train_features_path, val_features = self.val_features_path, 
                obs_len = self.obs_len, pred_len = self.pred_len, normalize = self.normalize, 
                use_delta = self.use_delta, use_map = self.use_map, use_social = self.use_social, 
                lr = self.lr, cfg = self.val_cfg, test = True, dataset = self.dataset)
        
        with open(self.val_cfg.gt, "rb") as f:
            gt_trajectories: Dict[int, np.ndarray] = pkl.load(f)

        with open(self.val_cfg.traj_save_path, "rb") as f:
            forecasted_trajectories: Dict[int, List[np.ndarray]] = pkl.load(f)

        with open(self.val_features_path, "rb") as f:
            features_df: pd.DataFrame = pkl.load(f)
        
        evaluation(gt_trajectories, forecasted_trajectories, features_df, self.val_cfg.prune_n_guesses, 
                self.val_cfg.n_cl, self.val_cfg.max_n_guesses, self.val_cfg.max_neighbors_cl, self.val_cfg.n_guesses_cl, 
                self.pred_len, self.val_cfg.miss_threshold, self.obs_len, 0, None)
        

