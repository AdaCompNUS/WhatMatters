import os
import sys
import copy
import time
import torch
import numpy as np
from scipy import sparse
from typing import List, Tuple, Dict
from importlib import import_module

from simulator.base_simulator import Simulator
from simulator.LaneGCN.utils import Logger, load_pretrain, gpu, to_long, from_numpy, collate_fn, ref_copy
from summit_map.summit_api.map_utils.summit_map_api import SummitMap

from multiprocessing import Process, Queue
from numba import jit

class LaneGCN(Simulator):

    def __init__(self, config: Dict):
        
        sys.path.append(os.path.join(sys.path[0],'simulator','LaneGCN'))
        model = import_module('lanegcn')
        self.config, _, self.net, _, _, _ = model.get_model()
        # load pretrain model
        ckpt_path = config.model.weight
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(self.net, ckpt["state_dict"])
        self.net.eval()
        self.am = SummitMap()
        
    
    def run(self):

        start = time.time()
        for i in range(100):
            np.random.seed(0)
            input = np.random.normal(loc = [150,200], scale = 1, size = (20,20,2))
            data_run= self.preprocess(input)
            data_run = collate_fn(data_run)
            start1 = time.time()
            import pdb;pdb.set_trace()
            with torch.no_grad():
                output = self.net(data_run)
                results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
            end1 = time.time()
            print('Inference Once Costs %f seconds' % (end1-start1))
        end = time.time()
        print('100 Scenario Costs %f seconds' % (end-start))
        return np.squeeze(np.array(results))
    
    def array2dict(self, trajectories: np.ndarray):

        start = time.time()
        data_dict = self.getTrajectoryFeatures(trajectories)
        end1 = time.time()
        print('Get Trajectory Costs %f seconds' % (end1-start))
        
        procs=list()
        queue = Queue()
        for data in data_dict:
            p = Process(target=self.getMapFeatures, args=(data, queue))
            procs.append(p)
            p.start()
        for i,_ in enumerate(procs):
            data_dict[i]['graph'] = queue.get()
        for p in procs:
            p.join()
        end2 = time.time()
        print('Get Map Features Costs %f seconds' % (end2-end1))
        for data in data_dict:
            graph = modify(to_long(gpu(from_numpy(data['graph']))), self.config['cross_dist'])
            data['graph'] = graph
        end3 = time.time()
        print('Modify Map Features Costs %f seconds' % (end3-end2))
        return data_dict


    def dict2run(self, data_dict: List[Dict]):
        data_run = []
        for data in data_dict:
            new_data = dict()
            for key in ['city', 'orig', 'gt_preds', 'has_preds', 'theta', 'rot', 'feats', 'ctrs', 'graph']:
                if key in data:
                    new_data[key] = ref_copy(data[key])
            data_run.append(new_data)
        return data_run


    def getTrajectoryFeatures(self, trajectories: np.ndarray):
        
        """  "idx", "city", "feats", "ctrs", "orig", "theta", "rot", "graph"  """ 
        data_dict = []
        
        for i in range(trajectories.shape[0]):

            data = dict()
            data['city'] = self.getCity(trajectories)
            data['idx'] = i

            df = copy.deepcopy(trajectories)
            df[[0, i]] = df[[i, 0]]
            
            orig = df[0][19].copy().astype(np.float32)

            pre = df[0][18] - orig
            theta = np.pi - np.arctan2(pre[1], pre[0])

            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)

            feats, ctrs,  = [], []

            for j in range(df.shape[0]):

                feat = np.zeros((20, 3), np.float32)
                feat[:, :2] = np.matmul(rot, (df[j]- orig.reshape(-1, 2)).T).T
                feat[:, 2] = 1.0

                x_min, x_max, y_min, y_max = self.config['pred_range']
                if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                    continue

                ctrs.append(feat[-1, :2].copy())
                feat[1:, :2] -= feat[:-1, :2]
                feat[0, :2] = 0
            
                feats.append(feat)

            feats = np.asarray(feats, np.float32)
            ctrs = np.asarray(ctrs, np.float32)

            data['feats'] = feats
            data['ctrs'] = ctrs
            data['orig'] = orig
            data['theta'] = theta
            data['rot'] = rot
            data_dict.append(data)
        
        return data_dict

    def getMapFeatures(self, data, queue):

        """Get a rectangle area defined by pred_range."""

        x_min, x_max, y_min, y_max = self.config['pred_range']
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius)
        lane_ids = copy.deepcopy(lane_ids)
        
        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane
            
        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1
            
            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))
            
            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))
            
        node_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)
        num_nodes = count
        
        pre, suc = dict(), dict()
        for key in ['u', 'v']:
            pre[key], suc[key] = [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]
            idcs = node_idcs[i]
            
            pre['u'] += idcs[1:]
            pre['v'] += idcs[:-1]
            if lane.predecessors is not None:
                for nbr_id in lane.predecessors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre['u'].append(idcs[0])
                        pre['v'].append(node_idcs[j][-1])
                    
            suc['u'] += idcs[:-1]
            suc['v'] += idcs[1:]
            if lane.successors is not None:
                for nbr_id in lane.successors:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc['u'].append(idcs[-1])
                        suc['v'].append(node_idcs[j][0])

        lane_idcs = []
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int64))
        lane_idcs = np.concatenate(lane_idcs, 0)

        pre_pairs, suc_pairs, left_pairs, right_pairs = [], [], [], []
        for i, lane_id in enumerate(lane_ids):
            lane = lanes[lane_id]

            nbr_ids = lane.predecessors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        pre_pairs.append([i, j])

            nbr_ids = lane.successors
            if nbr_ids is not None:
                for nbr_id in nbr_ids:
                    if nbr_id in lane_ids:
                        j = lane_ids.index(nbr_id)
                        suc_pairs.append([i, j])

            nbr_id = lane.l_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    left_pairs.append([i, j])

            nbr_id = lane.r_neighbor_id
            if nbr_id is not None:
                if nbr_id in lane_ids:
                    j = lane_ids.index(nbr_id)
                    right_pairs.append([i, j])
        pre_pairs = np.asarray(pre_pairs, np.int64)
        suc_pairs = np.asarray(suc_pairs, np.int64)
        left_pairs = np.asarray(left_pairs, np.int64)
        right_pairs = np.asarray(right_pairs, np.int64)
                    
        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['pre'] = [pre]
        graph['suc'] = [suc]
        graph['lane_idcs'] = lane_idcs
        graph['pre_pairs'] = pre_pairs
        graph['suc_pairs'] = suc_pairs
        graph['left_pairs'] = left_pairs
        graph['right_pairs'] = right_pairs
        
        for k1 in ['pre', 'suc']:
            for k2 in ['u', 'v']:
                graph[k1][0][k2] = np.asarray(graph[k1][0][k2], np.int64)
        
        for key in ['pre', 'suc']:
            graph[key] += dilated_nbrs(graph[key][0], graph['num_nodes'], self.config['num_scales'])
        
        queue.put(graph)
        # return graph
        



def modify(graph, cross_dist, cross_angle=None):
    left, right = dict(), dict()

    lane_idcs = graph['lane_idcs']
    num_nodes = len(lane_idcs)
    num_lanes = lane_idcs[-1].item() + 1

    dist = graph['ctrs'].unsqueeze(1) - graph['ctrs'].unsqueeze(0)
    dist = torch.sqrt((dist ** 2).sum(2))
    hi = torch.arange(num_nodes).long().to(dist.device).view(-1, 1).repeat(1, num_nodes).view(-1)
    wi = torch.arange(num_nodes).long().to(dist.device).view(1, -1).repeat(num_nodes, 1).view(-1)
    row_idcs = torch.arange(num_nodes).long().to(dist.device)

    if cross_angle is not None:
        f1 = graph['feats'][hi]
        f2 = graph['ctrs'][wi] - graph['ctrs'][hi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = t2 - t1
        m = dt > 2 * np.pi
        dt[m] = dt[m] - 2 * np.pi
        m = dt < -2 * np.pi
        dt[m] = dt[m] + 2 * np.pi
        mask = torch.logical_and(dt > 0, dt < config['cross_angle'])
        left_mask = mask.logical_not()
        mask = torch.logical_and(dt < 0, dt > -config['cross_angle'])
        right_mask = mask.logical_not()
    
    pre = graph['pre_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    suc = graph['suc_pairs'].new().float().resize_(num_lanes, num_lanes).zero_()
    
    if graph['pre_pairs'].numel():
        pre[graph['pre_pairs'][:, 0], graph['pre_pairs'][:, 1]] = 1
    if graph['suc_pairs'].numel():
        suc[graph['suc_pairs'][:, 0], graph['suc_pairs'][:, 1]] = 1

    pairs = graph['left_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        left_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        left_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            left_dist[hi[left_mask], wi[left_mask]] = 1e6

        min_dist, min_idcs = left_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        left['u'] = ui.cpu().numpy().astype(np.int16)
        left['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        left['u'] = np.zeros(0, np.int16)
        left['v'] = np.zeros(0, np.int16)
    
    
        
    pairs = graph['right_pairs']
    if len(pairs) > 0:
        mat = pairs.new().float().resize_(num_lanes, num_lanes).zero_()
        mat[pairs[:, 0], pairs[:, 1]] = 1
        mat = (torch.matmul(mat, pre) + torch.matmul(mat, suc) + mat) > 0.5

        right_dist = dist.clone()
        mask = mat[lane_idcs[hi], lane_idcs[wi]].logical_not()
        right_dist[hi[mask], wi[mask]] = 1e6
        if cross_angle is not None:
            right_dist[hi[right_mask], wi[right_mask]] = 1e6

        min_dist, min_idcs = right_dist.min(1)
        mask = min_dist < cross_dist
        ui = row_idcs[mask]
        vi = min_idcs[mask]
        f1 = graph['feats'][ui]
        f2 = graph['feats'][vi]
        t1 = torch.atan2(f1[:, 1], f1[:, 0])
        t2 = torch.atan2(f2[:, 1], f2[:, 0])
        dt = torch.abs(t1 - t2)
        m = dt > np.pi
        dt[m] = torch.abs(dt[m] - 2 * np.pi)
        m = dt < 0.25 * np.pi

        ui = ui[m]
        vi = vi[m]

        right['u'] = ui.cpu().numpy().astype(np.int16)
        right['v'] = vi.cpu().numpy().astype(np.int16)
    else:
        right['u'] = np.zeros(0, np.int16)
        right['v'] = np.zeros(0, np.int16)

    graph['left'] = left
    graph['right'] = right
    return graph

def dilated_nbrs(nbr, num_nodes, num_scales):
    data = np.ones(len(nbr['u']), np.bool)
    csr = sparse.csr_matrix((data, (nbr['u'], nbr['v'])), shape=(num_nodes, num_nodes))

    mat = csr
    nbrs = []
    for i in range(1, num_scales):
        mat = mat * mat

        nbr = dict()
        coo = mat.tocoo()
        nbr['u'] = coo.row.astype(np.int64)
        nbr['v'] = coo.col.astype(np.int64)
        nbrs.append(nbr)
    return nbrs