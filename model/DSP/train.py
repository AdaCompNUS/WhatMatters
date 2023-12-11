import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import faulthandler
from tqdm import tqdm
#
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader
#
from model.DSP.loader import Loader
from model.DSP.utils.logger import Logger
from model.DSP.utils.utils import AverageMeter, AverageMeterForDict, check_loss_abnormal, str2bool
from model.DSP.utils.utils import save_ckpt

def main(feature_dir, adv_cfg_path, model, loss,
        obs_len, pred_len, use_cuda, model_path, train_cfg, dataset):
    mode = 'train'
    faulthandler.enable()
    start_time = time.time()

    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str

    loader = Loader(mode, feature_dir, adv_cfg_path, model, loss, device, is_ddp=False)
    (train_set, val_set), net, loss_fn, optimizer, evaluator = loader.load()

    dl_train = DataLoader(train_set,
                          batch_size=train_cfg.train_batch_size,
                          shuffle=True,
                          num_workers=8,
                          collate_fn=train_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)
    dl_val = DataLoader(val_set,
                        batch_size=train_cfg.val_batch_size,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=val_set.collate_fn,
                        drop_last=True,
                        pin_memory=True)

    niter = 0
    best_metric_val = 10e2
    flag_metric = 'brier_fde_k'

    for epoch in range(train_cfg.train_epoches):
        # * Train
        epoch_start = time.time()
        train_loss_meter = AverageMeterForDict()
        train_eval_meter = AverageMeterForDict()
        net.train()
        for i, data in enumerate(tqdm(dl_train)):
            out = net(data)
            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            optimizer.zero_grad()
            loss_out['loss'].backward()
            if bool(train_cfg.clipping):
                torch.nn.utils.clip_grad_norm_(net.parameters(), train_cfg.clipping)
            lr = optimizer.step(epoch)

            train_loss_meter.update(loss_out)
            train_eval_meter.update(eval_out)
            niter += train_cfg.train_batch_size

        loss_avg = train_loss_meter.metrics['loss'].avg


        if (epoch + 1) % train_cfg.val_interval == 0:
            # * Validation
            with torch.no_grad():
                val_start = time.time()
                val_loss_meter = AverageMeterForDict()
                val_eval_meter = AverageMeterForDict()
                net.eval()
                for i, data in enumerate(tqdm(dl_val)):
                    out = net(data)
                    loss_out = loss_fn(out, data)

                    post_out = net.post_process(out)
                    eval_out = evaluator.evaluate(post_out, data)

                    val_loss_meter.update(loss_out)
                    val_eval_meter.update(eval_out)

                if (epoch >= train_cfg.train_epoches / 2):
                    if val_eval_meter.metrics[flag_metric].avg < best_metric_val:
                        model_name = date_str + '_{}_best.tar'.format(model)
                        middle_name = "model/DSP/saved_models_summit/" if dataset == "SummitDataset" else "model/DSP/saved_models/"
                        save_ckpt(net, optimizer, epoch, middle_name, model_name)
                        best_metric_val = val_eval_meter.metrics[flag_metric].avg
                        print('Save the model: {}, {}: {:.4}, epoch: {}'.format(
                            model_name, flag_metric, best_metric_val, epoch))

    # save trained model
    middle_name = "model/DSP/saved_models_summit/" if dataset == "SummitDataset" else "model/DSP/saved_models/"
    model_name = date_str + '_{}_epoch{}.tar'.format(model, train_cfg.train_epoches)
    save_ckpt(net, optimizer, epoch, middle_name, model_name)
    print('Save the model to {}'.format(middle_name + model_name))

    print('\nExit...')


if __name__ == "__main__":
    main()
