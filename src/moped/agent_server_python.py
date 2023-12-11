#!/home/phong/anaconda3/envs/lanegcn/bin/python


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
import time
import sys

import grpc
import agentinfo_pb2
import agentinfo_pb2_grpc
import numpy as np
from moped_implementation.planner_wrapper import PlannerWrapper
import multiprocessing

_THREAD_CONCURRENCY = multiprocessing.cpu_count()
MAX_HISTORY_MOTION_PREDICTION = 20
import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

class Greeter(agentinfo_pb2_grpc.MotionPredictionServicer):

    def __init__(self):
        self.planner = PlannerWrapper()

    def Predict(self, request, context):
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
        probs, predictions = self.planner.do_predictions(agents_history)

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

def serve():
    # Do compilation first
    #planner = PlannerWrapper()
    #trajectories = np.random.normal(loc=[50,200], scale=1, size=(20,20,2))
    #planner.do_predictions(trajectories) # probs shape (number_agents,) predictions shape (number_agents, pred_len, 2)
    
    server = grpc.server(futures.ProcessPoolExecutor(max_workers=1)) # Divide 8 because we run 4 parallely
    agentinfo_pb2_grpc.add_MotionPredictionServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(filename=f"{current_dir}/logfilexxx.txt",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    logging.info(sys.executable)
    serve()
