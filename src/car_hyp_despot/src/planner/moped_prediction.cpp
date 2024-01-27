//
// Created by cunjun on 25/8/22.
//

#include "moped_prediction.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;


MotionPredictionClient::MotionPredictionClient(std::shared_ptr<Channel> channel)
            : stub_(agentinfo::MotionPrediction::NewStub(channel)) {}

// Assembles the client's payload, sends it and presents the response back
// from the server.
std::map<int, std::vector<double>> MotionPredictionClient::Predict(std::vector<AgentStruct> neighborAgents, CarStruct car) {
        // Data we are sending to the server.
        agentinfo::PredictionRequest request;

        for (const AgentStruct &tempAgent : neighborAgents) {
            std::vector<COORD> hist = tempAgent.coordHistory.coord_history;
            agentinfo::AgentInfo *agentInfo = request.add_agentinfo();
            for (unsigned int i = 0; i < hist.size(); i++) {
                agentInfo->add_x(hist[i].x);
                agentInfo->add_y(hist[i].y);

            }
            agentInfo->set_agentid(tempAgent.id);
            agentInfo->set_agenttype(tempAgent.type);
        }

        // Adding car info
        std::vector<COORD> carHist = car.coordHistory.coord_history;
        agentinfo::AgentInfo *agentInfo = request.add_agentinfo();
        for (unsigned int i = 0; i < carHist.size(); i++) {
            agentInfo->add_x(carHist[i].x);
            agentInfo->add_y(carHist[i].y);
            //std::cout << " car x-y: " << carHist[i].x << " - " << carHist[i].y;
        }
        //std::cout << std::endl;
        agentInfo->set_agentid(-1);
        agentInfo->set_agenttype(AgentType::car);

        // Container for the data we expect from the server.
        agentinfo::PredictionResponse reply;

        // Context for the client. It could be used to convey extra information to
        // the server and/or tweak certain RPC behaviors.
        ClientContext context;

        // The actual RPC.
        Status status = stub_->Predict(&context, request, &reply);

        // Act upon its status.
        std::map<int, std::vector<double>> results;

        if (status.ok()) {
            //std::cout << "GRPC Call Success" << " size " << reply.agentinfo_size() << std::endl;
            for (int i = 0; i < reply.agentinfo_size(); i++) {
                std::vector<double> returnAgentInfo;

                agentinfo::ProbabilityInfo *probInfo = reply.mutable_probinfo(i);
                agentinfo::AgentInfo *agentInfo = reply.mutable_agentinfo(i);

                returnAgentInfo.push_back(probInfo->prob());
                returnAgentInfo.push_back(agentInfo->x(0));
                returnAgentInfo.push_back(agentInfo->y(0));

//                std::cout << "Prediction agent " << probInfo->agentid() << " with x_next: " << agentInfo->x(0)
//                << " and y_next: " << agentInfo->y(0)  << " prob: " << probInfo->prob() << std::endl;

                results.insert({agentInfo->agentid(), returnAgentInfo});
            }
            return results;
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
            std::cout << "RPC Failed!" << std::endl;
            return results;
        }
}

std::map<int, std::vector<double>> callPython(std::vector<AgentStruct> neighborAgents, CarStruct car) {
    // Instantiate the client. It requires a channel, out of which the actual RPCs
    // are created. This channel models a connection to an endpoint specified by
    // the argument "--target=" which is the only expected argument.
    // We indicate that the channel isn't authenticated (use of
    // InsecureChannelCredentials()).

    // static MotionPredictionClient mopedClient(
    //     grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));

    static MotionPredictionClient mopedClient(
        grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials()));
        
    return mopedClient.Predict(neighborAgents, car);
}
