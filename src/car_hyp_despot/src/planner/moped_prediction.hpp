//
// Created by cunjun on 25/8/22.
//

#ifndef P3_CATKIN_WS_NEW_MOPED_PREDICTION_HPP
#define P3_CATKIN_WS_NEW_MOPED_PREDICTION_HPP

#include <grpcpp/grpcpp.h>
#include "agentinfo.grpc.pb.h"
#include "state.h"
#include <vector>

class MotionPredictionClient {
public:
    MotionPredictionClient(std::shared_ptr<grpc::Channel> channel);
    std::map<int, std::vector<double>> Predict(std::vector<AgentStruct> neighborAgents, CarStruct car);
private:
    std::unique_ptr<agentinfo::MotionPrediction::Stub> stub_;
};

std::map<int, std::vector<double>> callPython(std::vector<AgentStruct> neighborAgents, CarStruct car);

#endif //P3_CATKIN_WS_NEW_MOPED_PREDICTION_HPP
