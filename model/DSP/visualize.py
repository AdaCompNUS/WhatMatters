import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from model.DSP.loader import Loader
from model.DSP.utils.logger import Logger
from model.DSP.utils.utils import AverageMeterForDict
from model.DSP.visualizers.visualizer_dsp import VisualizerDsp

def main(feature_dir, adv_cfg_path, model, loss,
        obs_len, pred_len, use_cuda, model_path, shuffle):
    
    mode = 'test'
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    if model == 'dsp':
        vis = VisualizerDsp()
    else:
        assert False, "Unknown visualizer"

    if not model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(model_path)

    # if mode != 'test':
    #     loader = Loader(mode, feature_dir, adv_cfg_path, model, loss, device, is_ddp=False)
    #     print('[Resume]Loading state_dict from {}'.format(model_path))
    #     loader.set_resmue(model_path)
    #     (train_set, val_set), net, _, _, _ = loader.load()
    #     net.eval()

    #     dl_train = DataLoader(train_set,
    #                           batch_size=1,
    #                           shuffle=shuffle,
    #                           num_workers=0,
    #                           collate_fn=train_set.collate_fn,
    #                           drop_last=False)
    #     dl_val = DataLoader(val_set,
    #                         batch_size=1,
    #                         shuffle=shuffle,
    #                         num_workers=0,
    #                         collate_fn=val_set.collate_fn,
    #                         drop_last=False)

    #     with torch.no_grad():
    #         for i, data in enumerate(tqdm(dl_val)):
    #             out = net(data)
    #             post_out = net.post_process(out)
    #             vis.draw_once(post_out, data, show_map=True)

    # else:
    #     # test
    loader = Loader(mode, feature_dir, adv_cfg_path, model, loss, device, is_ddp=False)
    print('[Resume]Loading state_dict from {}'.format(model_path))
    loader.set_resmue(model_path)
    test_set, net, _, _, _ = loader.load()
    net.eval()

    dl_test = DataLoader(test_set,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        collate_fn=test_set.collate_fn)
    preds = {}
    pred_probs = {}
    cities = {}
    for ii, data in enumerate(tqdm(dl_test)):
        with torch.no_grad():
            out = net(data)
            post_out = net.post_process(out)
            results = (torch.matmul(post_out["traj_pred"][:,:,:,:2].detach().cpu(), data['ROT'][0].T) + data['ORIG'][0]).numpy()
            probs = post_out["prob_pred"].detach().cpu().numpy()
        for i, (argo_idx, pred_traj, pred_prob) in enumerate(zip(data["SEQ_ID"], results, probs)):
            preds[argo_idx] = pred_traj.squeeze()
            pred_probs[argo_idx] = pred_prob.squeeze()
            cities[argo_idx] = data['CITY_NAME'][i]
    from argoverse.evaluation.competition_util import generate_forecasting_h5
    generate_forecasting_h5(preds, f"submit.h5", probabilities=pred_probs)  # this might take awhile


    print('\nExit...')


if __name__ == "__main__":
    main()
