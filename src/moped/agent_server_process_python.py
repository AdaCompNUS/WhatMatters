#!/home/phong/anaconda3/envs/HiVT/bin/python


# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""

from concurrent import futures
import logging
import logging.handlers
import time
import sys
import contextlib
import socket
import cProfile

import grpc
import agentinfo_pb2
import agentinfo_pb2_grpc
import numpy as np
from moped_implementation.planner_wrapper import PlannerWrapper
import multiprocessing

import torch

_THREAD_CONCURRENCY = multiprocessing.cpu_count()


MAX_HISTORY_MOTION_PREDICTION = 20
import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

class Greeter(agentinfo_pb2_grpc.MotionPredictionServicer):

    def __init__(self):
        self.planner = PlannerWrapper()

    def Predict(self, request, context):
        #print(f"Multiprocessing id: {multiprocessing.current_process()}")
        xy_pos_list = []
        agent_id_list = []
        start = time.time()
        for agentInfo in request.agentInfo:
            x_pos = np.array(agentInfo.x)
            y_pos = np.array(agentInfo.y)
            #logging.info(f"Shape of x_pos is {x_pos.shape} and y_pos is {y_pos.shape}")
            xy_pos = np.concatenate([x_pos[..., np.newaxis], y_pos[..., np.newaxis]], axis=1)  # shape (n,2)
            # first axis, we pad width=0 at beginning and pad width=MAX_HISTORY_MOTION_PREDICTION-xy_pos.shape[0] at the end
            # second axis (which is x and y axis), we do not pad anything as it does not make sense to pad anything
            xy_pos = np.pad(xy_pos, pad_width=((0, MAX_HISTORY_MOTION_PREDICTION - xy_pos.shape[0]), (0, 0)),
                            mode="edge")
            xy_pos_list.append(xy_pos)
            #logging.info(f"[P] Shape of x_pos is {x_pos.shape} and y_pos is {y_pos.shape} and after padd {xy_pos.shape}")

            agent_id_list.append(agentInfo.agentId)

        agents_history = np.stack(xy_pos_list)  # Shape (number_agents, observation_len, 2)


        #prediction_time = time.time()

        # Simple simulation

        # probs shape (number_agents,) predictions shape (number_agents, pred_len, 2)

        #logging.info(f"Inputs shape: {agents_history.shape}")
        try:
            probs, predictions = self.planner.do_predictions(agents_history)
        except Exception as e:
            logging.info(f"Error in prediction: {e} with inputs {agents_history}")
            probs = np.ones(agents_history.shape[0])
            # Predictions is the last known position but with shape (number_agents, 1, 2)
            predictions = agents_history[:, [-1], :]


        response_time = time.time()

        # Build response:
        response = agentinfo_pb2.PredictionResponse()
        for i, id in enumerate(agent_id_list):
            prob_info = agentinfo_pb2.ProbabilityInfo(prob = probs[i], agentId = id)
            agent_info = agentinfo_pb2.AgentInfo()
            agent_info.agentId = id
            agent_info.x.append(predictions[i][0][0]) # Get number_agent_id of 1st axis, of first pred of 2nd axis, of x of 3rd axis
            agent_info.y.append(predictions[i][0][1]) # Get number_agent_id of 1st axis, of first pred of 2nd axis, of y of 3rd axis

            response.agentInfo.append(agent_info)
            response.probInfo.append(prob_info)

        end = time.time()
        #logging.info(f"Time for producing response: {end - response_time}")
        #logging.info(f"Time for running end-to-end: {end - start}")

        return response


def _run_server(bind_address):
    print(f"Server started. Awaiting jobs...")
    NUMBER_THREADS = 1
    #NUMBER_THREADS = int(_THREAD_CONCURRENCY/4)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(NUMBER_THREADS)),
        options=[
            ("grpc.max_send_message_length", -1),
            ("grpc.max_receive_message_length", -1),
            ("grpc.so_reuseport", 1),
            ("grpc.use_local_subchannel_pool", 1),
        ],
    )
    agentinfo_pb2_grpc.add_MotionPredictionServicer_to_server(Greeter(), server)
    server.add_insecure_port(bind_address)
    server.start()
    server.wait_for_termination()


@contextlib.contextmanager
def _reserve_port():
    """Find and reserve a port for all subprocesses to use"""
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError("Failed to set SO_REUSEPORT.")
    sock.bind(("", 50056))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def main():
    """
    Inspired from https://github.com/grpc/grpc/blob/master/examples/python/multiprocessing/server.py
    """
    NUM_WORKERS = 1 # Each pomdp planner will call their own moped
    #NUM_WORKERS = int(_THREAD_CONCURRENCY/4) #When using 1 instance of server

    print(f"Initializing server with {NUM_WORKERS} workers")
    torch.multiprocessing.set_start_method('spawn')
    print(f"Number of processes before creating workers: {multiprocessing.active_children()}")

    with _reserve_port() as port:
        bind_address = f"[::]:{port}"
        print(f"Binding to {bind_address}")
        sys.stdout.flush()
        workers = []
        for _ in range(NUM_WORKERS):
            #worker = multiprocessing.Process(target=_run_server, args=(bind_address,))
            
            worker = torch.multiprocessing.Process(target=_run_server, args=(bind_address,))
            time.sleep(1) # Wait for 1 second to let moped knows what GPU to use
            worker.start()
            workers.append(worker)
            print(f"Number of processes after creating workers: {multiprocessing.active_children()}")

        for worker in workers:
            worker.join()


if __name__ == '__main__':
    max_bytes = 500 * 1024 * 1024  # 500Mb
    backup_count = 1  # keep only the latest file
    filename = f"{current_dir}/logfilexxx.txt"

    handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )

    formatter = logging.Formatter(
        '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        handlers=[handler],
        level=logging.DEBUG,
    )

    logging.info(sys.executable)

    main()
