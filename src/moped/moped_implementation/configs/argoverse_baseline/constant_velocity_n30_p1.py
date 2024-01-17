### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    train=dict(
        features = '/home/cunjun/moped_data/argoverse_features/forecasting_features_train.pkl',
        pipeline = []
    ),
    test=dict(
        features = '/home/cunjun/moped_data/argoverse_features/forecasting_features_test.pkl',
        pipeline=[]
    ),
    val=dict(
        features = '/home/cunjun/moped_data/argoverse_features/forecasting_features_val.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'ConstantVelocity',
    obs_len=20,
    pred_len=30,
    avg_points=[20],  # Number of predicted trajectories given past points. Must be at most history_points, at least 2
    train_cfg = dict(

    ), # ConstantVelocity no needs training
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/constant_velocity_n30_p6/'

_base_ = [
    '../_base_/eval.py'
]