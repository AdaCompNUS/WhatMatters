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
"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function

import logging

import grpc
import agentinfo_pb2
import agentinfo_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = agentinfo_pb2_grpc.MotionPredictionStub(channel)

        # Build request
        a= {10: {"x": [150,152,160], "y":[250.2, 252.2, 255.2], "type": 1},
            20: {"x": [151,159,165, 170, 180], "y":[255.2, 259.2, 260.2, 240, 245], "type": 1},
            3: {"x": [160.1, 166], "y": [256, 257], "type": 2}}

        prediction_request = agentinfo_pb2.PredictionRequest()
        for agent_id, agent_info in a.items():
            agentInfo = agentinfo_pb2.AgentInfo()
            agentInfo.agentId = agent_id
            agentInfo.agentType = agent_info["type"]
            agentInfo.x.extend(agent_info["x"])
            agentInfo.y.extend(agent_info["y"])

            prediction_request.agentInfo.append(agentInfo)


        response = stub.Predict(prediction_request)
        print(f"Greeter client received: {response}" )

        agentInfo = response.agentInfo
        probInfo = response.probInfo

        for i in range(len(agentInfo)):
            agent_info = agentInfo[i]
            prob_info = probInfo[i]
            print(f"At {i}, agent id {agent_info.agentId} pred_x {agent_info.x}"
                  f" pred_y {agent_info.y} and prob {prob_info.prob}")


if __name__ == '__main__':
    logging.basicConfig()
    run()
