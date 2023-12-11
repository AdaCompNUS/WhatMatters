
#ifndef MOPEDPARAMS_H
#define MOPEDPARAMS_H

namespace MopedParams {
    const int MAX_HISTORY_MOTION_PREDICTION = 20;
    const bool PHONG_DEBUG = false; // to print my logging so that I can understand the logic
    const bool PHONG_ESSENTIAL_DEBUG = true; // to print logging for benchmarking number of simulations, iterations and time
    // of motion predictions
    const bool PHONG_REWARD_DEBUG = true; // to print reward of each step
    const bool PHONG_DESPOT_DEBUG = false; // to print despot code so I understand pomdp planning
    const bool USE_MOPED = true; // true if using motion prediction instead of original code
}

#endif