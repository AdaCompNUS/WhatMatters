import argparse
import os
import shutil
import sys
import logging
from typing import Any
from mmcv import Config
from utils.logger import Logger
from utils.summit_dataloader import SummitBaselineDataset


logging.basicConfig(format='%(levelname)s-%(message)s', level=logging.DEBUG)
printer = logging.getLogger('train')

def parse_args():
    parser = argparse.ArgumentParser(description="Training a motion prediction model")

    parser.add_argument('config', help='training config file')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--work-dir',
                        help= 'the directory to save the file containing training metrics and checkpoints')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed training)')


    args = parser.parse_args()

    return args

def check_args(args: Any) -> bool:
    work_dir = args.work_dir
    checkpoint = args.resume_from

    return True

def construct_model(cfg, logger):
    # To avoid python version conflicts library, we only import the model based on the name
    model_name = cfg.model.type
    model = None
    if model_name == "ConstantVelocity":
        from model.argoverse_baseline.constant_velocity import ConstantVelocity
        model = ConstantVelocity(config=cfg, logger=logger)
    if model_name == "ConstantAcceleration":
        from model.argoverse_baseline.constant_acceleration import ConstantAcceleration
        model = ConstantAcceleration(config=cfg, logger=logger)
    if model_name == "KNearestNeighbor":
        from model.argoverse_baseline.knearest_neighbor import KNearestNeighbor
        model = KNearestNeighbor(config=cfg, logger=logger)
    if model_name == "SummitKNearestNeighbor":
        from model.argoverse_baseline.summit_knn import SummitKNearestNeighbor
        model = SummitKNearestNeighbor(config=cfg, logger=logger)
    if model_name == "Gamma":
        from model.gamma.gamma_wrapper import Gamma
        model = Gamma(config=cfg, logger=logger)
    if model_name == "LSTM":
        from model.LSTM.lstm_wrapper import LSTM
        model = LSTM(config=cfg, logger=logger)
    if model_name == "LaneGCN":
        from model.LaneGCN.lanegcn_wrapper import LaneGCN
        model = LaneGCN(config=cfg, logger=logger)
    if model_name == "HiVT":
        from model.HiVT.hivt_wrapper import HiVT
        model = HiVT(config=cfg, logger=logger)
    if model_name == "DSP":
        from model.DSP.DSP_wrapper import DSP
        model = DSP(config=cfg, logger=logger)
    if model_name == "HOME":
        from model.HOME.src.home.home_wrapper import HOME
        model = HOME(config=cfg, logger=logger)
    return model

def construct_dataset(config):
    '''
    This method check builds dataset from raw_dir and save pickle file at features_path
    cfg: Config class
    '''
    if config.data.dataset_type == "SummitDataset" and config.model.type not in ["LaneGCN", "HiVT", "HOME"]:
        obs_len = config.model.obs_len
        pred_len = config.model.pred_len
        raw_train_dir = config.data.train.raw_dir
        raw_val_dir = config.data.val.raw_dir
        raw_test_dir = config.data.test.raw_dir
        processed_dir_train = config.data.train.processed_dir
        processed_dir_val = config.data.val.processed_dir
        processed_dir_test = config.data.test.processed_dir
        
        # Step 1. Check raw dir of each train/test/val and process
        dataloader = SummitBaselineDataset(split="train", obs_len= obs_len,
                                     pred_len = pred_len,
                                     raw_train_dir = raw_train_dir,
                                     raw_val_dir = raw_val_dir,
                                     raw_test_dir = raw_test_dir,
                                     processed_dir=processed_dir_train)

        # Step 2. We store merged files to features (overwrite any features file beforehand)
        dataloader.merge_saved_features(config.data.train.features)
        print(f"Done processing {raw_train_dir} and store in {config.data.train.features}")

        # Step 3. Do the same for val
        dataloader = SummitBaselineDataset(split="val", obs_len=obs_len,
                                               pred_len=pred_len,
                                               raw_train_dir=raw_train_dir,
                                               raw_val_dir=raw_val_dir,
                                               raw_test_dir=raw_test_dir,
                                               processed_dir=processed_dir_val)
        dataloader.merge_saved_features(config.data.val.features)
        print(f"Done processing {raw_val_dir} and store in {config.data.val.features}")

        # Use 'val' for 'testmode' so that it contains candidate centerlines
        dataloader = SummitBaselineDataset(split="test", obs_len=obs_len,
                                               pred_len=pred_len,
                                               raw_train_dir=raw_train_dir,
                                               raw_val_dir=raw_val_dir,
                                               raw_test_dir=raw_test_dir,
                                               processed_dir=processed_dir_test)
        dataloader.merge_saved_features(config.data.test.features)
        print(f"Done processing {raw_test_dir} in test mode and store in {config.data.test.features}")

    elif config.data.dataset_type == "ArgoverseDataset" and config.model.type not in ["LaneGCN", "HiVT", "HOME"]:
        # For ArgoverseDataset, the file must exists for features
        # If no test features, run python compute_features.py --data_dir ../datasets/val/data/ --feature_dir ../features/forecasting_features/ --mode test --name val_testmode
        assert os.path.isfile(config.data.train.features), f"No train features file at {config.data.train.features}"
        assert os.path.isfile(config.data.val.features), f"No val features file at {config.data.val.features}"
        assert os.path.isfile(config.data.test.features), f"No test features file at {config.data.test.features}"


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)

    if not check_args(args):
        return

    # Initialize logger path
    exp_name = cfg.filename.split('/')[-1].split('.')[0]
    if args.work_dir:
        work_dir = args.work_dir
    else: # if not work_dir, default is in work_dirs/config_file/yyyy-mm-dd_hh-mm-ss_config_file/
        work_dir = cfg.work_dir

    logger = Logger(work_dir=work_dir, exp_name=exp_name, hyperparams=cfg, mode="train", printer=printer)

    printer.info(f"Training {cfg.model.type}\n"
          f"Loading configuration files f{args.config}\n")

    if args.resume_from:
        printer.info(f'Resuming training from {args.resum_frome}')

    construct_dataset(cfg)
    model = construct_model(cfg, logger)
    model.train()

if __name__ == "__main__":
    main()
