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

import time
import math
from typing import Any, Dict, List, Tuple, Union

import argparse
from mmcv import Config
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import copy

from utils.lstm_utils import ModelUtils, LSTMDataset
import utils.baseline_utils as baseline_utils

from simulator.base_simulator import Simulator
from utils.social_features_vectorize import SocialFeaturesUtilsVectorize
from pathlib import Path
ROOT = Path(__file__).resolve().parent


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

@torch.jit.interface
class ModuleInterface(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor: # `input` has a same name in Sequential forward
        pass

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
        submodule: ModuleInterface = self.lstm1
        hidden = submodule(embedded, hidden)
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

class LSTM(Simulator):
    WEIGHT_PATH_DEFAULT = f'{ROOT}/lstm/LSTM_rollout30.pth.tar' # must be default or nomap-nosocial
    WEIGHT_PATH_SOCIAL = f'{ROOT}/lstm_social/LSTM_rollout30.pth.tar' # must be default or nomap-nosocial
    def __init__(self, use_social = True, gpu_device = 0):
        self.social_features_utils_vectorize_instance = SocialFeaturesUtilsVectorize()
        self.normalize = True # Support True/False. From inputs to process
        self.use_delta = True # Support True/False. From inputs to process
        self.use_map = False # Do not support True
        self.use_social = use_social # Support True and False. For building inputs
        if self.use_social:
            WEIGHT = LSTM.WEIGHT_PATH_SOCIAL
        else:
            WEIGHT = LSTM.WEIGHT_PATH_DEFAULT
        self.input_size = 5 if self.use_social else 2
        self.encoder = EncoderRNN(input_size = self.input_size)
        self.decoder = DecoderRNN(output_size=2)

        print("LSTM Using GPU device: old {} with new {}".format(gpu_device, torch.cuda.current_device()))
        self.gpu_device = torch.cuda.current_device()
        torch.cuda.set_device(self.gpu_device)
        print("LSTM Using GPU device: {}".format(self.gpu_device))

        #self.gpu_device = torch.device("cuda:{}".format(gpu_device) if torch.cuda.is_available() else "cpu")
        #torch.cuda.set_device(gpu_device)
        #print(f"Using device {self.gpu_device}")
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)
        print(f"Before model device {self.encoder}")

        self.encoder.to(self.gpu_device)
        self.decoder.to(self.gpu_device)
        print(f"After model device {self.encoder}")

        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=0.001)
        self.model_utils = ModelUtils()
        self.model_utils.load_checkpoint(WEIGHT, self.encoder, self.decoder, encoder_optimizer,decoder_optimizer)
        self.encoder.eval()
        self.decoder.eval()

    def run(self, trajectories: np.ndarray, pred_len: int):
        translation, rotation, reference, normalize_traj_arr = self.preprocess(trajectories)
        normalize_traj_arr = normalize_traj_arr.astype(float)
        _input = torch.tensor(normalize_traj_arr, dtype=torch.float32).to(self.gpu_device)
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        # Initialize encoder hidden state
        encoder_hidden = self.model_utils.init_hidden(
            batch_size,
            self.encoder.module.hidden_size if self.use_cuda else self.encoder.hidden_size)

        # Encode observed trajectory, use [10,input_length] to decrease encoder time

        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = self.encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = add_noise(encoder_hidden)

        decoder_outputs = torch.zeros(
            (batch_size, 30, 2)).to(self.gpu_device)

        # Decode hidden state in future trajectory
        for di in range(pred_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                        decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs as next step
            decoder_input = decoder_output
            
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(reference)
        abs_helpers["TRANSLATION"] = np.array(translation)
        abs_helpers["ROTATION"] = np.array(rotation)
        args = Config(dict(
                    use_map=self.use_map,
                    normalize=self.normalize,
                    use_delta=self.use_delta,
                    pred_len=30,
                    joblib_batch_size = 100,
                ))
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
                _input.clone().cpu().numpy(),
                decoder_outputs.detach().clone().cpu().numpy(),
                args,
                abs_helpers,
            )

        return abs_outputs[:,:pred_len,:]

    def array2dict(self, trajectories: np.ndarray):
        pass

    def dict2run(self, data_dict: dict):
        pass

    def get_relative_distance(self, data: np.ndarray, mode: str, obs_len: int, pred_len: int) -> np.ndarray:
        """Convert absolute distance to relative distance in place and return the reference (first value).
        Args:
            data (numpy array): Data array of shape (num_tracks x seq_len X num_features). Distances are always the first 2 features
            mode: train/val/test
            args: Arguments passed to the baseline code
        Returns:
            reference (numpy array): First value of the sequence of data with shape (num_tracks x 2). For map based baselines, it will be first n-t distance of the trajectory.
        """
        reference = copy.deepcopy(data[:, 0, :2])

        if mode == "test":
            traj_len = obs_len
        else:
            traj_len = obs_len + pred_len

        for i in range(traj_len - 1, 0, -1):
            data[:, i, :2] = data[:, i, :2] - data[:, i - 1, :2]
        data[:, 0, :] = 0
        return reference

    def normalize_trajectories(self, trajectories: np.ndarray, obs_len: int):
        """

        Args:
            trajectories: shape [num_agents, obs_len, 2]

        Returns:
            translation [num_agents]
            , rotations [num_agents, 6]
            , and normalized trajectories [num_agents, obs_len, 2]

        """
        translation = []
        rotation = []

        normalized_traj = []
        x_coord_seq = trajectories[:, :, 0]
        y_coord_seq = trajectories[:, :, 1]

        # Normalize each trajectory
        for i in range(x_coord_seq.shape[0]):
            xy_seq = np.stack((x_coord_seq[i], y_coord_seq[i]), axis=-1)

            start = xy_seq[0]

            # First apply translation
            m = [1, 0, 0, 1, -start[0], -start[1]]
            ls = LineString(xy_seq)

            # Now apply rotation, taking care of edge cases
            ls_offset = affine_transform(ls, m)
            end = ls_offset.coords[obs_len - 1]
            if end[0] == 0 and end[1] == 0:
                angle = 0.0
            elif end[0] == 0:
                angle = -90.0 if end[1] > 0 else 90.0
            elif end[1] == 0:
                angle = 0.0 if end[0] > 0 else 180.0
            else:
                angle = math.degrees(math.atan(end[1] / end[0]))
                if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
                    angle = -angle
                else:
                    angle = 180.0 - angle

            # Rotate the trajetory
            ls_rotate = rotate(ls_offset, angle, origin=(0, 0)).coords[:]

            # Normalized trajectory
            norm_xy = np.array(ls_rotate)

            # Update the containers
            normalized_traj.append(norm_xy)
            translation.append(m)
            rotation.append(angle)

        # Update the dataframe and return the normalized trajectory
        normalize_traj_arr = np.stack(normalized_traj)

        return translation, rotation, normalize_traj_arr

    def build_all_agents_vec(self, trajectories: np.ndarray):
        """

        Args:
            trajectories: shape [num_agents, obs_len, 2]

        Returns:
            merged_vec: shape [num_agents, obs_len, 5]
        """

        if self.use_social:
            social_features = []
            social_index = self.social_features_utils_vectorize_instance.filter_tracks(trajectories, 20, {"X": 0, "Y": 1})
            for agent_index in range(trajectories.shape[0]):
                social_index_i = [item for item in social_index if item != agent_index]
                social_feature = self.social_features_utils_vectorize_instance.compute_social_features_vec(
                    trajectories, social_index_i, agent_index, 20
                )
                social_features.append(social_feature)
            ## shape [num_agents, obs_len, 3], where 3 is [min_dist_from, min_dist_back, num_neighbors]
            social_features = np.array(social_features)
            # shape [num_agents, obs_len, 5], where 5 is [x, y, min_dist_from, min_dist_back, num_neighbors]
            merged_features = np.concatenate([trajectories, social_features], axis=2)
        else:
            # shape [num_agents, obs_len, 5], where 5 is [x, y, None, None, None]
            merged_features = np.full((trajectories.shape[0], trajectories.shape[1], 5), None)
            merged_features[:, :, 0:2] = trajectories

        return merged_features
    
    def preprocess(self, input_features: np.ndarray):
        """
            Processing the input features.
            "social": ["X", "Y", "MIN_DISTANCE_FRONT", "MIN_DISTANCE_BACK", "NUM_NEIGHBORS"]
            "none": ["X", "Y"]

            Output:
            "social": ["X", "Y"]
            "none": ["X", "Y"]

            Args:
                input_features: shape [num_agents, obs_len, 2]

            Returns:
                translation,
                rotation,
                reference,
                input_features: shape [num_agents, obs_len, 5]
        """
        # Step 1. Build agent vector
        # Output shape [num_agents, obs_len, 5]
        a = time.time()
        agent_vec = self.build_all_agents_vec(input_features)

        b = time.time()
        # Step 2. Normalize if normalize = True
        if self.normalize:
            xy_trajectories = agent_vec[:, :, 0:2] # the first 2 columns are x, y
            translation, rotation, normalize_traj_arr = self.normalize_trajectories(xy_trajectories, obs_len=20)
            # Add back the social features
            normalize_traj_arr = np.concatenate((normalize_traj_arr, agent_vec[:, :, 2:]), axis=-1)
        else:
            translation, rotation, normalize_traj_arr = None, None, agent_vec

        # Step 2.a. Dependings on if self.social, filter the columns
        if not self.use_social:
            normalize_traj_arr = normalize_traj_arr[:, :, 0:2]

        c = time.time()
        # Step 3. Get relative distance. xy must be in first 2 columns
        if self.use_delta:
            reference = self.get_relative_distance(normalize_traj_arr, mode="test", obs_len=20, pred_len=30)
        else:
            reference = None

        d = time.time()
        #print("Time for each step: ", b-a, c-b, d-c, d-a)

        return translation, rotation, reference, normalize_traj_arr



