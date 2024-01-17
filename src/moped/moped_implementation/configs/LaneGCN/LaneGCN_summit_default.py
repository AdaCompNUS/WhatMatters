### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'SummitDataset',
    joblib_batch_size=100,
    train=dict(
        raw_dir = '../datasets_summit/summit_10HZ/train/data', # raw datafiles
        features = '../features/forecasting_features/train_summit_10HZ_crs_dist6_angle90.p',
        pipeline = []
    ),
    test=dict(
        raw_dir='../datasets_summit/summit_3hz/val_hivt/data',
        features='../features/forecasting_features/val_summit_3hz_crs_dist6_angle90.p',
        pipeline=[]
    ),
    val=dict(
        raw_dir='../datasets_summit/summit_3hz/val_hivt/data',
        features='../features/forecasting_features/val_summit_3hz_crs_dist6_angle90.p',
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
        weight = 'simulator/LaneGCN/36.000.ckpt',
        resume = ''
    ),
    val_cfg=dict(
        weight = 'simulator/LaneGCN/36.000.ckpt',
        resume = ''
    )
)



### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LaneGCN_default/'

_base_ = [
    '../_base_/eval.py'
]