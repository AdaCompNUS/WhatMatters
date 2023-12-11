"""lstm_train_test.py runs the LSTM baselines training/inference on forecasting dataset.

Note: The training code for these baselines is covered under the patent <PATENT_LINK>.

Example usage:
python lstm_train_test.py
    --model_path saved_models/lstm.pth.tar
    --test_features ../data/forecasting_data_test.pkl
    --train_features ../data/forecasting_data_train.pkl
    --val_features ../data/forecasting_data_val.pkl
    --use_delta --normalize
"""

import os
import shutil
import tempfile
import time
import random
from typing import Any, Dict, List, Tuple, Union

import argparse
from mmcv import Config
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.baseline_config as config
from utils.lstm_utils import ModelUtils, LSTMDataset
import utils.baseline_utils as baseline_utils

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best_loss = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]
noise_dim=(8,)

def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)

def add_noise(encoder_hidden):
    noise_shape = (encoder_hidden[0].shape[0],) + noise_dim

    h_decoder = get_noise(noise_shape, "gaussian")
    c_decoder = get_noise(noise_shape, "gaussian")

    _h_list = torch.cat((encoder_hidden[0],h_decoder), dim=1)
    _c_list = torch.cat((encoder_hidden[1],c_decoder), dim=1)

    return (_h_list, _c_list)

class EncoderRNN(nn.Module):
    """Encoder Network."""
    def __init__(self,
                 input_size: int = 2,
                 embedding_size: int = 8,
                 hidden_size: int = 16):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""
    def __init__(self, embedding_size=8, hidden_size=24, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


def train(
        train_loader: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        rollout_len: int = 30,
) -> None:
    """Train the lstm network.

    Args:
        train_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        rollout_len: current prediction horizon

    """
    global global_step

    for i, (_input, target, helpers) in enumerate(train_loader):
        _input = _input.to(device)
        target = target.to(device)

        # Set to train mode
        encoder.train()
        decoder.train()

        # Zero the gradients
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize losses
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state add gaussian noise
        decoder_hidden = add_noise(encoder_hidden)

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(rollout_len):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
                                                     
            
            decoder_outputs[:, di, :] = decoder_output 
            
            # Update loss
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / rollout_len

        # Backpropagate
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        if global_step % 1000 == 0:

            # Log results
            print(
                f"Train -- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

        global_step += 1


def validate(
        val_loader: Any,
        epoch: int,
        criterion: Any,
        encoder: Any,
        decoder: Any,
        encoder_optimizer: Any,
        decoder_optimizer: Any,
        model_utils: ModelUtils,
        prev_loss: float,
        use_map: bool,
        use_social: bool,
        decrement_counter: int,
        rollout_len: int = 30,
        dataset: str = "ArgoverseDataset",
) -> Tuple[float, int]:
    """Validate the lstm network.

    Args:
        val_loader: DataLoader for the train set
        epoch: epoch number
        criterion: Loss criterion
        encoder: Encoder network instance
        decoder: Decoder network instance
        encoder_optimizer: optimizer for the encoder network
        decoder_optimizer: optimizer for the decoder network
        model_utils: instance for ModelUtils class
        prev_loss: Loss in the previous validation run
        decrement_counter: keeping track of the number of consecutive times loss increased in the current rollout
        rollout_len: current prediction horizon

    """
    global best_loss
    total_loss = []
    
    for i, (_input, target, helpers) in enumerate(val_loader):

        _input = _input.to(device)
        target = target.to(device)

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        output_length = target.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size,
            encoder.module.hidden_size if use_cuda else encoder.hidden_size)

        # Initialize loss
        loss = 0

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state add gaussian noise
        decoder_hidden = add_noise(encoder_hidden)

        decoder_outputs = torch.zeros(target.shape).to(device)

        # Decode hidden state in future trajectory
        for di in range(output_length):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Update losses for all benchmarks
            loss += criterion(decoder_output[:, :2], target[:, di, :2])

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get average loss for pred_len
        loss = loss / output_length
        total_loss.append(loss)

        if i % 10 == 0:

            cprint(
                f"Val -- Epoch:{epoch}, loss:{loss}, Rollout: {rollout_len}",
                color="green",
            )

    # Save
    val_loss = sum(total_loss) / len(total_loss)
    
    middle_dir = "saved_models_summit" if dataset == "SummitDataset" else "saved_models"
    print("val_loss:", val_loss)
    print("best_loss:", best_loss)
    if val_loss <= best_loss:
        best_loss = val_loss
        if use_social:
            save_dir = "model/LSTM/" + middle_dir + "/lstm_social"
        else:
            save_dir = "model/LSTM/" + middle_dir + "/lstm"
        print("Save Model to", save_dir)

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            },
        )

    # Keep track of the loss to change preiction horizon
    if val_loss <= prev_loss:
        decrement_counter = 0
    else:
        decrement_counter += 1

    return val_loss, decrement_counter


def infer_absolute(
        test_loader: torch.utils.data.DataLoader,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        start_idx: int,
        forecasted_save_dir: str,
        model_utils: ModelUtils,
        pred_len,
        use_map,
        use_delta,
        normalize,
        cfg,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    forecasted_trajectories = {}
    for i, (_input, target, helpers) in enumerate(test_loader):

        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # get max_n_guesses trajectories 
        num_candidates = cfg.max_n_guesses
        abs_outputs_multi = []
        for i in range(num_candidates):
            
            # Initialize encoder hidden state
            encoder_hidden = model_utils.init_hidden(
                batch_size,
                encoder.module.hidden_size if use_cuda else encoder.hidden_size)

            # Encode observed trajectory
            for ei in range(input_length):
                encoder_input = _input[:, ei, :]
                encoder_hidden = encoder(encoder_input, encoder_hidden)

            # Initialize decoder input with last coordinate in encoder
            decoder_input = encoder_input[:, :2]

            # Initialize decoder hidden state as encoder hidden state
            decoder_hidden = add_noise(encoder_hidden)

            decoder_outputs = torch.zeros(
                (batch_size, pred_len, 2)).to(device)

            # Decode hidden state in future trajectory
            for di in range(pred_len):
                decoder_output, decoder_hidden = decoder(decoder_input,
                                                        decoder_hidden)
                decoder_outputs[:, di, :] = decoder_output

                # Use own predictions as inputs at next step
                decoder_input = decoder_output

            # Get absolute trajectory
            abs_helpers = {}
            abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"]) #(100, 2)
            abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"]) #(100, 6)
            abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"]) #(100,)
            args = Config(dict(
                    use_map=use_map,
                    normalize=normalize,
                    use_delta=use_delta,
                    pred_len=pred_len,
                    joblib_batch_size = cfg.joblib_batch_size,
                ))
            
            abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
                _input.clone().cpu().numpy(),
                decoder_outputs.detach().clone().cpu().numpy(),
                args,
                abs_helpers,
            )
            abs_outputs_multi.append(abs_outputs)

        for i in range(abs_outputs_multi[0].shape[0]):
            seq_id = helpers_dict["SEQ_PATHS"][i]
            forecasted_trajectory = []
            for j in range(num_candidates):
                forecasted_trajectory.append(abs_outputs_multi[j][i])
            forecasted_trajectories[seq_id] = forecasted_trajectory

    with open(os.path.join(forecasted_save_dir, f"{start_idx}.pkl"),
              "wb") as f:
        pkl.dump(forecasted_trajectories, f)

def infer_helper(
        curr_data_dict: Dict[str, Any],
        start_idx: int,
        encoder: EncoderRNN,
        decoder: DecoderRNN,
        model_utils: ModelUtils,
        forecasted_save_dir: str,
        obs_len,
        pred_len,
        use_map,
        use_delta,
        use_social,
        normalize,
        cfg,
):
    """Run inference on the current joblib batch.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    curr_test_dataset = LSTMDataset(curr_data_dict, normalize, use_delta, use_map, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=cfg.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    print(f"#### LSTM+social inference at {start_idx} ####"
            ) if use_social else print(
                f"#### LSTM inference at {start_idx} ####")
    infer_absolute(
        curr_test_loader,
        encoder,
        decoder,
        start_idx,
        forecasted_save_dir,
        model_utils,
        pred_len,
        use_map,
        use_delta,
        normalize,
        cfg,
    )


def main(train_features, val_features, obs_len, pred_len, normalize, use_delta, use_map, use_social, lr, cfg, test, dataset):
    """Main."""
    global best_loss
    args = Config(dict(
        use_map=use_map,
        use_social=use_social,
        normalize=normalize,
        use_delta=use_delta,
        obs_len=obs_len,
        pred_len=pred_len,
    ))

    if not baseline_utils.validate_args(args):
        return

    print(f"Using all ({joblib.cpu_count()}) CPUs....")
    if use_cuda:
        print(f"Using all ({torch.cuda.device_count()}) GPUs...")

    model_utils = ModelUtils()

    # key for getting feature set
    # Get features
    if use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    # Get data
    if dataset == "SummitDataset" and test == False:

        args = Config(dict(
            use_map=use_map,
            use_social=use_social,
            normalize=normalize,
            use_delta=use_delta,
            obs_len=obs_len,
            pred_len=pred_len,
            train_features=train_features,
            test_features=None,
            val_features=val_features
        ))
        data_dict = baseline_utils.get_data(args, baseline_key)


    elif dataset == "SummitDataset" and test == True:
        args = Config(dict(
            use_map=use_map,
            use_social=use_social,
            normalize=normalize,
            use_delta=use_delta,
            obs_len=obs_len,
            pred_len=pred_len,
            train_features=None,
            test_features=val_features,
            val_features=None
        ))
        data_dict = baseline_utils.get_data(args, baseline_key)

    elif test == False:
        args = Config(dict(
            use_map=use_map,
            use_social=use_social,
            normalize=normalize,
            use_delta=use_delta,
            obs_len=obs_len,
            pred_len=pred_len,
            train_features=train_features,
            test_features=None,
            val_features=val_features
        ))
        data_dict = baseline_utils.get_data(args, baseline_key)

    else:
        args = Config(dict(
            use_map=use_map,
            use_social=use_social,
            normalize=normalize,
            use_delta=use_delta,
            obs_len=obs_len,
            pred_len=pred_len,
            train_features=None,
            test_features=val_features,  # Swap val for test
            val_features=None
        ))
        data_dict = baseline_utils.get_data(args, baseline_key)

    # Get model
    criterion = nn.MSELoss()
    encoder = EncoderRNN(
        input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
    decoder = DecoderRNN(output_size=2)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    # If model_path provided, resume from saved checkpoint
    if cfg.model_path is not None and os.path.isfile(cfg.model_path):
        epoch, rollout_len, _ = model_utils.load_checkpoint(
            cfg.model_path, encoder, decoder, encoder_optimizer,
            decoder_optimizer)
        start_epoch = epoch + 1
        start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1

    else:
        start_epoch = 0
        start_rollout_idx = 0

    if not test:
        # Get PyTorch Dataset
        train_dataset = LSTMDataset(data_dict, normalize, use_delta, use_map, "train")
        val_dataset = LSTMDataset(data_dict, normalize, use_delta, use_map, "val")

        # Setting Dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=model_utils.my_collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val_batch_size,
            drop_last=False,
            shuffle=False,
            collate_fn=model_utils.my_collate_fn,
        )

        print("Training begins ...")

        decrement_counter = 0

        epoch = start_epoch
        global_start_time = time.time()
        for i in range(start_rollout_idx, len(ROLLOUT_LENS)):
            rollout_len = ROLLOUT_LENS[i]
            best_loss = float("inf")
            prev_loss = best_loss
            while epoch < cfg.end_epoch:
                start = time.time()
                train(
                    train_loader,
                    epoch,
                    criterion,
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    model_utils,
                    rollout_len,
                )
                end = time.time()

                print(
                    f"Training epoch completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                )

                epoch += 1
                if epoch % 5 == 0:
                    start = time.time()
                    prev_loss, decrement_counter = validate(
                        val_loader,
                        epoch,
                        criterion,
                        encoder,
                        decoder,
                        encoder_optimizer,
                        decoder_optimizer,
                        model_utils,
                        prev_loss,
                        use_map,
                        use_social,
                        decrement_counter,
                        rollout_len,
                        dataset,
                    )
                    end = time.time()
                    print(
                        f"Validation completed in {(end - start) / 60.0} mins, Total time: {(end - global_start_time) / 60.0} mins"
                    )

                    # If val loss increased 3 times consecutively, go to next rollout length
                    if decrement_counter > 2:
                        break

    else:

        start_time = time.time()

        temp_save_dir = tempfile.mkdtemp()

        test_size = data_dict["test_input"].shape[0]
        args = Config(dict(
            joblib_batch_size = cfg.joblib_batch_size
        ))
        test_data_subsets = baseline_utils.get_test_data_dict_subset(
            data_dict,args)

        # test_batch_size should be lesser than joblib_batch_size
        Parallel(n_jobs=10, verbose=2)(
            delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
                                  model_utils, temp_save_dir, obs_len, pred_len, use_map,
                                  use_delta, use_social, normalize, cfg)
            for i in range(0, test_size, cfg.joblib_batch_size))

        baseline_utils.merge_saved_traj(temp_save_dir, cfg.traj_save_path)
        shutil.rmtree(temp_save_dir)

        end = time.time()
        print(f"Test completed in {(end - start_time) / 60.0} mins")
        print(f"Forecasted Trajectories saved at {cfg.traj_save_path}")



