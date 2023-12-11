### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    train=dict(
        features = '../features/forecasting_features/forecasting_features_train.pkl',
        pipeline = []
    ),
    test=dict(
        features = '../features/forecasting_features/forecasting_features_val_testmode.pkl',
        pipeline=[]
    ),
    val=dict(
        features = '../features/forecasting_features/forecasting_features_val.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'ConstantAcceleration',
    obs_len=20,
    pred_len=30,
    avg_points=[20, 15, 10, 8, 6, 3],  # Number of predicted trajectories given past points. Must be at most history_points-2, at least 3
    train_cfg = dict(

    ),
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/constant_acceleration_n30_p6/'

_base_ = [
    '../_base_/eval.py'
]