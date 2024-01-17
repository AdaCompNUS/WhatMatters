### ----------- Dataset settings --------------- ###

data = dict(

    train=dict(
        pipeline = []
    ),
    test=dict(
        pipeline = []
    ),
    val=dict(
        pipeline = []
    ),
)

### -----------  Model -----------  ###

model = dict(
    type = 'LaneGCN',
    obs_len=20,
    pred_len=30,
    k=6, # number of trajectories
    weight = '../results/lanegcn_summit/36.000.ckpt'
)




### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LaneGCN_simulator_default/'

_base_ = [
    '../_base_/eval.py'
]