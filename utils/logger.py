"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""

import time
import os.path as osp
import os
import atexit
from utils.common_utils import colorize, convert_json
import shutil
import json
import joblib
import random
import string
from mmcv import Config
from typing import Dict, List, Any
import numpy as np
import yaml
import pickle as pkl

WORKING_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # absolute director of moped


class Logger():
    '''
    Logger to save the models, information and logging statistics
    Steps to save are:
    1. logger.save_config(locals()) to store all the configuration (should sit a first line in training script)
    2. logger.save_scaler() to store data's scaler
    3. logger.setup_tf_saver(tf_model) to setup the tf model
    4. For epoch:
            for mini-batch:
                logger.store() to store each stats of batch
            logger.save_state() to save the tf model
            logger.log_tabular() to calculate values stored by logger.store()
                                and print the neccesary info and save
                                into a progress temp dict
            logger.dump_tabular() to flush the progress temp dict into file
    '''
    def __init__(self, work_dir: str, exp_name:str, hyperparams: Config, mode: str, printer):
        '''
        Initialize the output directory
        :param work_dir: the folder to contain running results
        :param exp_name: the current algorithm
        :param hyperparams: hyper parames to setup the output_dir
        :param mode: the running mode 'train', 'test' or 'val'
        '''

        self.mode = mode
        self.printer = printer

        if mode == 'train':
            #1. Set up exp_name, output_dir, output_file
            # exp_name
            assert exp_name is not None, "You must provide the exp_name"
            self.exp_name = exp_name
            # suffix + output_dir

            self.output_dir = self.setup_output_dir(work_dir=work_dir)

            if osp.exists(self.output_dir):
                self.printer.warning("Warning: {0} exists. Storing data into this folder anyway.".format(self.output_dir))
            else:
                os.makedirs(self.output_dir)
                self.printer.info("Creating {0}. Storing the progress file in this folder.".format(self.output_dir))

            # output_file
            output_fname = 'progress.txt'
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            self.printer.info(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))

            #2. Set up log headers
            self.first_row=True
            self.log_headers = []
            self.log_current_row = {} # Store value of current epoch

            # 3. Set up hypermeters
            self.printer.info(colorize('Saving config:\n', color='cyan', bold=True))
            hyperparams.dump(os.path.join(self.output_dir, exp_name + ".py"))

            if 'gamma' in exp_name: # we need to also dump yaml for gamma to read parameters
                with open('config.yml', 'w') as outfile:
                    yaml.dump(hyperparams, outfile, default_flow_style=False)

            self.train_log_dir = osp.join(self.output_dir, 'train')
            os.makedirs(self.train_log_dir, exist_ok=True)
        else:
            self.output_dir = work_dir
            os.makedirs(self.output_dir, exist_ok=True)
            self.printer.info(f"Storing data into folder {self.output_dir}")

            self.val_log_dir = osp.join(self.output_dir, 'val')
            os.makedirs(self.val_log_dir, exist_ok=True)

            self.test_log_dir = osp.join(self.output_dir, 'test')
            os.makedirs(self.test_log_dir, exist_ok=True)


    def get_test_log_dir(self):
        return self.test_log_dir

    def get_val_log_dir(self):
        return self.val_log_dir

    def get_train_log_dir(self):
        return self.train_log_dir

    def setup_output_dir(self, work_dir: str, datestamp=True):
        '''
        We follow spinningup logging name
        Config the output_dir = data_dir/exp_name/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]

        :param suffix:
        :param data_dir:
        :param datestamp:
        :return:
        '''
        hms_time = time.strftime("%Y-%m-%d_%H-%M-%S") if datestamp else ""

        # if the last basename of work_dir is same as exp_name, then we do not need to join
        if self.exp_name in work_dir:
            data_dir = work_dir
        else:
            data_dir = osp.join(work_dir, self.exp_name)

        #3. [inner_prefix]exp_name[suffix]
        subfolder = ''.join([hms_time, '_', self.exp_name])

        data_dir = osp.join(data_dir, subfolder)

        return data_dir

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        self.printer.info(colorize(msg, color, bold=True))

    def store(self):
        '''
        Do nothing here
        :return:
        '''
        pass

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = Logger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
        self.printer.info(colorize('Saving config:\n', color='cyan', bold=True))
        self.printer.info(output)
        with open(osp.join(self.output_dir, 'config.txt'), "w") as f:
            f.write(output)


    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration (epoch).
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def dump_tabular(self):
        '''
        Write all of the diagnostics from the current iteration.
        Call this once only after  finishing running an epoch
        Writes both to stdout, and to the output file.
        :return:
        '''
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        max_val_len = 15
        print_fmt = f'| {{:>{max_key_len}}} | {{:>{max_val_len}}} |'
        n_slashes = 22 + max_key_len
        print("*" * n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = f"{val:8.3g}" if hasattr(val, "__float__") else f"{val}"
            self.printer.info(print_fmt.format(key, valstr))
            vals.append(val)
        self.printer.info('*' * n_slashes, flush=True)
        if self.output_file is not None:
            max_key_len = max(key_lens) # Reassign for flushing to file
            if self.first_row:
                key_lines = [f'{key:{max_key_len+2}.{max_key_len}}' for key in self.log_headers]
                self.output_file.write(''.join(key_lines)+'\n')
            val_lines = [f'{val:<{max_key_len+2}.5g}' if hasattr(val,'__float__') else f'{val:<{max_key_len+2}}' for val in vals] # add 2 to easy to see
            self.output_file.write(''.join(val_lines)+'\n')
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False

    def get_subdir_for_dump(self, mode=None):
        if mode == None:
            mode = self.mode
        if mode == "train":
            log_dir = self.train_log_dir
        elif mode == "test":
            log_dir = self.test_log_dir
        elif mode == "val":
            log_dir = self.val_log_dir
        else:
            log_dir = self.output_dir
        return log_dir

    def dump_forecasted_trajectories(self, forecasted_trajectories: Dict[int, List[np.ndarray]],
                                     forecasted_probs: Dict[int, List[np.ndarray]],
                                     forecasted_trajectories_filename : str = "prediction_trajectories.pkl",
                                     forecasted_probs_filename: str= "prediction_probs.pkl", mode=None) -> None:

        log_dir = self.get_subdir_for_dump(mode)

        if 'pkl' not in forecasted_trajectories_filename:
            forecasted_trajectories_filename = forecasted_trajectories_filename.split('.')[0] + '.pkl'
        else:
            if mode != None:
                forecasted_trajectories_filename = forecasted_trajectories_filename.split('.')[0] + f'_{mode}.pkl'

        if 'pkl' not in forecasted_probs_filename:
            forecasted_probs_filename = forecasted_probs_filename.split('.')[0] + '.pkl'
        else:
            if mode != None:
                forecasted_probs_filename = forecasted_probs_filename.split('.')[0] + f'_{mode}.pkl'

        forecasted_trajectories_path = os.path.join(log_dir, forecasted_trajectories_filename)

        with open(forecasted_trajectories_path, "wb") as f:
            pkl.dump(forecasted_trajectories, f)

        self.printer.info(f'Done Dumping {forecasted_trajectories_filename} at {forecasted_trajectories_path}')

        forecasted_probs_path = os.path.join(log_dir, forecasted_probs_filename)

        with open(forecasted_probs_path, "wb") as f:
            pkl.dump(forecasted_probs, f)

        self.printer.info(f'Done Dumping {forecasted_probs_filename} at {forecasted_probs_path}')

    def dump_groundtruth_trajectory(self, groundtruth_trajectory: Dict[int, np.ndarray],
                                    filename = "groundtruth_trajectory.pkl", mode=None) -> None:

        log_dir = self.get_subdir_for_dump(mode)

        if 'pkl' not in filename:
            filename = filename.split('.')[0] + '.pkl'
        else:
            if mode != None:
                filename = filename.split('.')[0] + f'_{mode}.pkl'

        path = os.path.join(log_dir, filename)

        with open(path, "wb") as f:
            pkl.dump(groundtruth_trajectory, f)

        self.printer.info(f'Done Dumping {filename} at {path}')

    def temp_dum_gamma(self, trajectory: List[np.ndarray], file_name: str):
        log_dir = self.get_subdir_for_dump()
        path = os.path.join(log_dir, file_name)
        with open(path, "wb") as f:
            pkl.dump(trajectory, f)

    def dump_pkl_model(self, model: Any, model_path: str):
        model_path = os.path.join(self.get_subdir_for_dump("train"), model_path)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with open(model_path, "wb") as f:
            pkl.dump(model, f)

        print(f"Done dumping at {model_path}")

    def info(self, x):
        self.printer.info(x)

    def warning(self, x):
        self.printer.warning(x)

