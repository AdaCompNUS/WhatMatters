import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from model.DSP.loader import Loader
from model.DSP.utils.utils import AverageMeter, AverageMeterForDict, str2bool




def main(feature_dir, adv_cfg_path, model, loss,
        obs_len, pred_len, use_cuda, model_path, val_batch_size):
    mode = 'val'
    faulthandler.enable()

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(model_path)

    loader = Loader(mode, feature_dir, adv_cfg_path, model, loss, device, is_ddp=False)
    print('[Resume]Loading state_dict from {}'.format(model_path))
    loader.set_resmue(model_path)
    (train_set, val_set), net, _, _, evaluator = loader.load()

    dl_val = DataLoader(val_set,
                        batch_size=val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=True)

    net.eval()

    with torch.no_grad():
        # * Validation
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val)):
            out = net(data)
            post_out = net.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

        print('\nValidation set finish, cost {} secs'.format(time.time() - val_start))
        print('-- ' + val_eval_meter.get_info())

    print('\nExit...')


if __name__ == "__main__":
    main()
