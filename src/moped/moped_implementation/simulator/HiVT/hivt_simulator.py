

import time
import copy
import torch
import numpy as np
from typing import List, Tuple, Dict

from itertools import permutations
from itertools import product
import pytorch_lightning as pl

from torch_geometric.data import Batch
from simulator.HiVT.models.hivt import HiVT as model
from simulator.HiVT.utils import TemporalData, from_numpy
from simulator.base_simulator import Simulator

from summit_map.summit_api.map_utils.summit_map_api import SummitMap

from pathlib import Path
ROOT = Path(__file__).resolve().parent
print(ROOT)

class HiVT(Simulator):
    WEIGHT_PATH = f'{ROOT}/epoch=63-step=28671.ckpt'
    def __init__(self, gpu_device: int = 0):

        # Select a random GPU device among 1,2,3:
        #self.gpu_device = torch.device("cuda:{}".format(gpu_device) if torch.cuda.is_available() else "cpu")

        # Fix ordinal bug
        print("Using GPU device: old {} with new {}".format(gpu_device, torch.cuda.current_device()))
        self.gpu_device = torch.cuda.current_device()
        torch.cuda.set_device(self.gpu_device)
        print("Using GPU device: {}".format(self.gpu_device))

        ckpt_path = HiVT.WEIGHT_PATH
        self.model = model.load_from_checkpoint(checkpoint_path=ckpt_path, parallel=True).cuda()
        self.am = SummitMap()
    
    # def run(self):
    #     np.random.seed(0)
    #     start = time.time()
    #
    #     for _ in range(100):
    #         input = np.random.normal(loc = [150,200], scale = 1, size = (20,20,2))
    #         data_run= self.preprocess(input)
    #         output, p = self.model(data_run)
    #         results = output[:,:,:,:2].permute(1,0,2,3).detach().numpy()
    #
    #     end = time.time()
    #     print('100 Scenario Costs %f seconds' % (end-start))
    #     print(results.shape)
    #     return results
    #     # (Pdb) results.shape
    #     # (20, 6, 30, 2)

    def run(self, obs_trajectory):
        data_run= self.preprocess(obs_trajectory).cuda()
        #print(f"Device of self.model is {self.model.device}")
        with torch.no_grad():
            # GPU mode:
            output, p = self.model(data_run)
            #output = output.cpu()
            #data_run = data_run.cpu()
            # CPU mode:
            #output, p = self.model(data_run)

            #origin_all = np.expand_dims(data_run['positions'][:, 19], [0, 2])

            local_origin = torch.unsqueeze(torch.unsqueeze(data_run['positions'][:, 19], 0), 2)

            local_angle = torch.tensor([[[torch.cos(data_run['rotate_angles'][i]), torch.sin(data_run['rotate_angles'][i])], 
                                         [-torch.sin(data_run['rotate_angles'][i]), torch.cos(data_run['rotate_angles'][i])]] for i in range(local_origin.shape[1])], device=self.gpu_device)

            global_origin = data_run['origin']
            global_angle = torch.tensor([[torch.cos(data_run.theta), torch.sin(data_run.theta)],
                                         [-torch.sin(data_run.theta), torch.cos(data_run.theta)]], device=self.gpu_device)

            results = torch.matmul(output[:, :, :, :2], local_angle)  + local_origin
            results = torch.matmul(results, global_angle) + global_origin

            results = results.permute(1,0,2,3).detach().cpu().numpy()

        return results


    def array2dict(self, trajectories: np.ndarray):

        kwargs = self.process_sequence(trajectories, self.am, self.model.hparams.local_radius)
        data_dict = TemporalData(**kwargs)

        return data_dict

    def dict2run(self, data_dict: Dict):

        data_dict = from_numpy(data_dict)
        data_run = Batch.from_data_list([data_dict])

        return data_run

    def process_sequence(self,
                        trajectories: np.ndarray,
                        am: SummitMap,
                        radius: float) -> Dict:

        # filter out actors that are unseen during the historical time steps
        df = copy.deepcopy(trajectories)

        agent_index = 0
        av_index = len(df)-1
        agent_df = df[agent_index]
        av_df = df[av_index]

        city = self.getCity(df)
        num_nodes = df.shape[0]

        # make the scene centered at AV
        origin = torch.tensor([av_df[-1][0], av_df[-1][1]], dtype=torch.float) # Using last agent as ego
        av_heading_vector = origin - torch.tensor([av_df[-2][0], av_df[-2][1]], dtype=torch.float)
        theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]])
        
        # initialization
        x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool) # False means need predict, all agent is False in simulator
        bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)

        for actor_id, actor_df in enumerate(df):
            node_idx = actor_id
            node_steps = [timestamp for timestamp in range(20)]
            padding_mask[node_idx, node_steps] = False
            xy = torch.from_numpy(actor_df).float()
            x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
            heading_vector = x[node_idx, 19] - x[node_idx, 18]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])

        # bos_mask is True if time step t is valid and time step t-1 is invalid
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

        positions = x.clone()
        x[:, 20:] = 0
        x[:, 1: 20] = x[:, 1: 20] - x[:, : 19]
        x[:, 0] = torch.zeros(num_nodes, 2)

        # get lane features at the current time step
        df_19 = np.array([pos[19] for pos in df])
        node_inds_19 = [node for node in range(num_nodes)]
        node_positions_19 = torch.from_numpy(df_19).float()
        (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
        lane_actor_vectors) = self.get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)

        y = None
        seq_id = 0

        return {
            'x': x[:, : 20],  # [N, 20, 2]
            'positions': positions,  # [N, 50, 2]
            'edge_index': edge_index,  # [2, N x N - 1]
            'y': y,  # [N, 30, 2]
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,  # [N, 50]
            'bos_mask': bos_mask,  # [N, 20]
            'rotate_angles': rotate_angles,  # [N]
            'lane_vectors': lane_vectors,  # [L, 2]
            'is_intersections': is_intersections,  # [L]
            'turn_directions': turn_directions,  # [L]
            'traffic_controls': traffic_controls,  # [L]
            'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
            'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
            'seq_id': seq_id,
            'av_index': av_index,
            'agent_index': agent_index,
            'city': city,
            'origin': origin.unsqueeze(0),
            'theta': theta,
        }


    def get_lane_features(self,
                        am: SummitMap,
                        node_inds: List[int],
                        node_positions: torch.Tensor,
                        origin: torch.Tensor,
                        rotate_mat: torch.Tensor,
                        city: str,
                        radius: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                torch.Tensor]:
        lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
        lane_ids = set()
        for node_position in node_positions:
            lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
        node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
        for lane_id in lane_ids:
            lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
            lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
            is_intersection = am.lane_is_in_intersection(lane_id, city)
            turn_direction = am.get_lane_turn_direction(lane_id, city)
            traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
            lane_positions.append(lane_centerline[:-1])
            lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
            count = len(lane_centerline) - 1
            is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
            if turn_direction == 'NONE':
                turn_direction = 0
            elif turn_direction == 'LEFT':
                turn_direction = 1
            elif turn_direction == 'RIGHT':
                turn_direction = 2
            else:
                raise ValueError('turn direction is not valid')
            turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
            traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
        lane_positions = torch.cat(lane_positions, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)
        is_intersections = torch.cat(is_intersections, dim=0)
        turn_directions = torch.cat(turn_directions, dim=0)
        traffic_controls = torch.cat(traffic_controls, dim=0)

        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
        lane_actor_vectors = \
            lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]

        return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors


