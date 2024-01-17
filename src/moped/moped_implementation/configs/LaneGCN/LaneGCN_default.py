### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    joblib_batch_size=100,
    train=dict(
        features = '../features/forecasting_features/train_crs_dist6_angle90.p',
        raw_dir = '../datasets/train/data', # raw datafiles
        pipeline = []
    ),
    test=dict(
        features = '../features/forecasting_features/test_test.p',
        raw_dir='../datasets/test_obs/data',
        pipeline=[]
    ),
    val=dict(
        features = '../features/forecasting_features/val_crs_dist6_angle90.p',
        raw_dir= '../datasets/val/data',
        pipeline=[]
    ),
)

### -----------  Model -----------  ###

model = dict(
    type = 'LaneGCN',
    obs_len=20,
    pred_len=30,
    k=6, # number of trajectories
    train_cfg = dict(
        weight = '',  # use which weight
        resume = ''   # train continue from which checkpoint path
    ),
    test_cfg = dict(
        weight = '../results/lanegcn/36.000.ckpt',
        resume = ''
    ),
    val_cfg=dict(
        weight = '../results/lanegcn/36.000.ckpt',
        resume = ''
    )
)



### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LaneGCN_default/'

_base_ = [
    '../_base_/eval.py'
]