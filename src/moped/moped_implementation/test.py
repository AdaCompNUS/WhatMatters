import argparse
from mmcv import Config
import pandas as pd
import os
import pickle
import logging
from utils.logger import Logger
from train import construct_model, construct_dataset

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.DEBUG)
printer = logging.getLogger('test')

def parse_args():
    parser = argparse.ArgumentParser(description="Generating testing results of a motion prediction model")

    parser.add_argument('config', help='validation config file')
    parser.add_argument('--work-dir',
                        help= 'the directory to save the file containing evaluation metrics')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--test', action="store_true",
                        help='if true, test, otherwise validate. Testing means to generate prediction'
                        'for upload to competition. Validation generates prediction and evaluate on ground-true')
    parser.add_argument('--viz', action="store_true",
                        help='if true, visualize the sequence')
    parser.add_argument('--viz_id_list',nargs='+', type=int, # Call like this (no comma between id): --viz_id_list 1 2 3
                        help='list of seq_id for visualization if viz is True. Default is all seq_id')

    args = parser.parse_args()

    return args

def get_ground_truth(cfg):
    df = pd.read_pickle(cfg.data.val.features)
    save_file = "../features/ground_truth_data"
    save_path = save_file + "/ground_truth_SummitDataset_3hz_val.pkl"
    if os.path.exists(save_path):
        return

    if not os.path.exists(save_file):
        os.makedirs(save_file)

    val_gt = {}
    for i in range(len(df)):
        seq_id = df.iloc[i]['SEQUENCE']
        curr_arr = df.iloc[i]['FEATURES'][20:][:, 3:5]
        val_gt[seq_id] = curr_arr

    with open(save_path, 'wb') as f:
        pickle.dump(val_gt, f)

def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict({
        "checkpoint": args.checkpoint
    })
    exp_name = cfg.filename.split('/')[-1].split('.')[0]
    if args.work_dir:
        work_dir = args.work_dir
    else:  # if not work_dir, default is in work_dirs/config_file/yyyy-mm-dd_hh-mm-ss_config_file/
        work_dir = cfg.work_dir

    logger = Logger(work_dir=work_dir, exp_name=exp_name, hyperparams=cfg, mode="test" if args.test else "val", printer=printer)
    printer.info(f"{'Testing to generate files for competition uploads' if args.test else 'Validating to get metrics'}\n"
          f"Loading configuration files f{args.config}\n")

    construct_dataset(cfg)
    model = construct_model(cfg, logger)
    if cfg.model.type == "LSTM":
        get_ground_truth(cfg)

    if args.test:
        model.test()
    else:
        model.validate()

    # Calculate the metrics and visualize
    # if not args.test and args.viz:
    #     viz_predictions_helper(forecasted_trajectories, groundtruth_trajectories,
    #                                features_df, obs_len, args.viz_id_list)


if __name__ == "__main__":
    main()
