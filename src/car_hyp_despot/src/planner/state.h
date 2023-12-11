#ifndef AGENT_STATE_H
#define AGENT_STATE_H
#include <vector>
#include <utility>

#include "coord.h"
#include "param.h"
#include "path.h"
#include "utils.h"
#include "despot/interface/pomdp.h"
#include "moped_param.h"

using namespace std;

using namespace despot;

enum AgentType { car=0, ped=1, num_values=2};

struct CoordHistory {
    std::vector<COORD> coord_history;
    int MAX_HISTORY = MopedParams::MAX_HISTORY_MOTION_PREDICTION;
    int true_history;
    double last_update_time; // time of the last update. We allow Add only if update_time - last_update_time > time_per_move
    CoordHistory() {
        true_history = 0;
        last_update_time = 0;
    }
    std::vector<double> time_history;

    void Add(COORD coord, double update_time, double time_per_move) {
        if ( (coord_history.size() > 0) && (update_time - last_update_time) < 0.95*time_per_move) {
            return;
        }
        if (coord_history.size() < MAX_HISTORY) {
            coord_history.push_back(coord);
            time_history.push_back(update_time);
        } else {
            coord_history.erase(coord_history.begin());
            time_history.erase(time_history.begin());
            coord_history.push_back(coord);
            time_history.push_back(update_time);
        }
        true_history += 1;
        last_update_time = update_time;
    }

    void CoordText(std::ostream& out) const {
        out << "Hist: " << coord_history.size() << ", true hist: " << true_history << " ";
        if (coord_history.size() >= 1) {
            out << ". 1st coord x:" << coord_history.at(0).x << " y:" << coord_history.at(0).y << " t:" << time_history.at(0) << " ";
        }
        if (coord_history.size() >= 2) {
            out << ". 2nd coord x:" << coord_history.at(1).x << " y:" << coord_history.at(1).y << " t:" << time_history.at(1) << " ";
        }
        if (coord_history.size() >= 4) {
            out << ". bef-last coord x:" << coord_history.at(coord_history.size()-2).x << " y:" 
            << coord_history.at(coord_history.size()-2).y << " t:" << time_history.at(coord_history.size()-2) << " ";
            out << ". last coord x:" << coord_history.at(coord_history.size()-1).x << " y:" 
            << coord_history.at(coord_history.size()-1).y << " t:" << time_history.at(coord_history.size()-1) << " ";
        }
    }

};

struct AgentStruct {

    AgentStruct(){
        set_default_values();
    }

    AgentStruct(COORD a, int b, int c) {
        set_default_values();
        pos = a;
        intention = b;
        id = c;
    }

    AgentStruct(COORD a, int b, int c, float _speed) {
        set_default_values();
        pos = a;
        intention = b;
        id = c;
        speed = _speed;
    }

    void set_default_values(){
        intention = -1;
        id = -1;
        speed = 0.0;
        mode = 1; //PED_DIS
        type = AgentType::car;
        pos_along_path = 0;
        bb_extent_x = 0;
        bb_extent_y = 0;
        heading_dir = 0;
        cross_dir = 0;
    }

    CoordHistory coordHistory;
    COORD pos;
    int mode;
    int intention; // intended path
    int pos_along_path; // traveled distance along the path
    int cross_dir;
    int id;
    AgentType type;
    double speed;
    COORD vel;
    double heading_dir;
    double bb_extent_x, bb_extent_y;

    void Text(std::ostream& out) const {
        out << "agent: id / pos / speed / vel / intention / dist2car / infront =  "
            << id << " / "
            << "(" << pos.x << ", " << pos.y << ") / "
            << speed << " / "
            << "(" << vel.x << ", " << vel.y << ") / "
            << intention << " / "
            << " (mode) " << mode
            << " (type) " << type
            << " (bb) " << bb_extent_x
            << " " << bb_extent_y
            << " (cross) " << cross_dir
            << " (heading) " << heading_dir << endl;
    }

    void ShortText(std::ostream& out) const {
        out << "Agent: id / type / pos / heading / vel: "
            << id << " / "
            << type << " / "
            << pos.x << "," << pos.y << " / "
            << heading_dir << " / "
            << vel.x << "," << vel.y << endl;
    }

    void PhongAgentText(std::ostream& out) const {
        out << " Agent: id " << std::dec << id << " " << ", cur pos: " << pos.x << ", " << pos.y << ". ";
        coordHistory.CoordText(out);
        out << endl;
    }
};

struct CarStruct {
    CoordHistory coordHistory;
    COORD pos;
    double vel;
    double heading_dir;/*[0, 2*PI) heading direction with respect to the world X axis */

    void PhongCarText(std::ostream& out) const {
        out << "Car pos: " << pos.x << ", " << pos.y << ". Car coords history: ";
        coordHistory.CoordText(out);
        out << endl;
    }
};

class PomdpState : public State {
public:
    CarStruct car;
    int num;
    AgentStruct agents[ModelParams::N_PED_IN];

    float time_stamp;

    PomdpState() {time_stamp = -1; num = 0;}

    string Text() const {
        return concat(car.vel);
    }

    void PomdpStateText(std::ostream& out) {
        out << "CarStruct address: " << &car << ". Car Info: ";
        car.PhongCarText(out);
        for (int i = 0; i < num; i++) {
            AgentStruct agent = agents[i];
            agent.PhongAgentText(out);
        }
        out << endl;
    }
};

class PomdpStateWorld : public State {
public:
    CarStruct car;
    int num;
    AgentStruct agents[ModelParams::N_PED_WORLD];

    float time_stamp;

    PomdpStateWorld() {time_stamp = -1; num = 0;}

    string Text() const {
        return concat(car.vel);
    }

    void assign(PomdpStateWorld& src){
        logi << "[PHONG] assign PomdpStateWorld" << endl;
        car.pos = src.car.pos;
        car.vel = src.car.vel;
        car.heading_dir = src.car.heading_dir;
        car.coordHistory = src.car.coordHistory;
        num = src.num;
        for (int i = 0; i < num; i++){
            agents[i].pos = src.agents[i].pos;
            agents[i].coordHistory = src.agents[i].coordHistory;
            agents[i].mode = src.agents[i].mode;
            agents[i].intention = src.agents[i].intention;
            agents[i].pos_along_path = src.agents[i].pos_along_path;
            agents[i].cross_dir = src.agents[i].cross_dir;
            agents[i].id = src.agents[i].id;
            agents[i].type = src.agents[i].type;
            agents[i].speed = src.agents[i].speed;
            agents[i].vel = src.agents[i].vel;
            agents[i].bb_extent_x = src.agents[i].bb_extent_x;
            agents[i].bb_extent_y = src.agents[i].bb_extent_y;
            agents[i].heading_dir = src.agents[i].heading_dir;
        }
        time_stamp = src.time_stamp;
    }

    void PomdpStateWorldText(std::ostream& out) {
        out << "CarStruct address: " << &car << ". Car Info: ";
        car.PhongCarText(out);
        for (int i = 0; i < num; i++) { // +2 is just to make sure that over than nums is not
            AgentStruct agent = agents[i];
            agent.PhongAgentText(out);
        }
        out << endl;
    }
};

#endif
