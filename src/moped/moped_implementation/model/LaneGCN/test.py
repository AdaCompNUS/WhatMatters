# ---------------------------------------------------------------------------
# Learning Lane Graph Representations for Motion Forecasting
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Ming Liang, Yun Chen
# ---------------------------------------------------------------------------

import argparse
import os
os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import pickle
import sys
from importlib import import_module

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model.LaneGCN.data import ArgoTestDataset
from model.LaneGCN.utils import Logger, load_pretrain


root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)



def main(train_dir, val_dir, test_dir, split, obs_len, pred_len, raw_train_dir, raw_val_dir, raw_test_dir,cfg):
    # Import all settings for experiment.
    model = import_module('lanegcn')
    config, _, collate_fn, net, loss, post_process, opt = model.get_model()
    config["train_split"] = train_dir   
    config["val_split"] = val_dir
    config["test_split"] = test_dir
    config["preprocess_train"] = raw_train_dir
    config["preprocess_val"] = raw_val_dir
    config['preprocess_test'] = raw_test_dir
    # load pretrain model
    ckpt_path = cfg.weight
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_pretrain(net, ckpt["state_dict"])
    net.eval()

    # Data loader for evaluation
    dataset = ArgoTestDataset(split, config, train=False)
    data_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )

    # begin inference
    preds = {}
    gts = {}
    cities = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
        for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
            preds[argo_idx] = pred_traj.squeeze()
            cities[argo_idx] = data["city"][i]
            gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    # save for further visualization
    res = dict(
        preds = preds,
        gts = gts,
        cities = cities,
    )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    
    # evaluate or submit
    if split == "val":
        # for val set: compute metric
        from argoverse.evaluation.eval_forecasting import (
            compute_forecasting_metrics,
        )
        # Max #guesses (K): 6
        _ = compute_forecasting_metrics(preds, gts, cities, 6, pred_len, 2)
        # Max #guesses (K): 1
        _ = compute_forecasting_metrics(preds, gts, cities, 1, pred_len, 2)
    else:
        # for test set: save as h5 for submission in evaluation server
        from argoverse.evaluation.competition_util import generate_forecasting_h5
        generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile

