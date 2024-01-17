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
    type = 'HiVT',
    obs_len=20,
    pred_len=30,
    k=6, # number of trajectories
    val_batch_size = 32,
    # ckpt_path = 'model/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt',
    ckpt_path = 'model/HiVT/checkpoints/HiVT-128/checkpoints_Summit/epoch=63-step=28671.ckpt',
)




### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/HiVT_simulator_default/'

_base_ = [
    '../_base_/eval.py'
]