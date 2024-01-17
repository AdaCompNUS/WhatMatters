### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'SummitDataset',
    train=dict(
        raw_dir='../datasets_summit/summit_sparse/train',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_sparse/train/train_features.pkl',
        # raw_dir='../datasets_summit/summit/train',
        # processed_dir='knn_processed',
        # features='../datasets_summit/summit/train/train_features.pkl',
        pipeline=[]
    ),
    test=dict(
        raw_dir = '../datasets_summit/summit_sparse/val',
        processed_dir='knn_processed_testmode',
        features='../datasets_summit/summit_sparse/val/val_features_testmode.pkl',
        # raw_dir = '../datasets_summit/summit_sparse/val',
        # processed_dir='knn_processed_testmode',
        # features='../datasets_summit/summit_sparse/val/val_features_testmode.pkl',
        # raw_dir = '../datasets_summit/summit/val',
        # processed_dir='knn_processed_testmode',
        # features='../datasets_summit/summit/val/val_features_testmode.pkl',
        pipeline=[]
    ),
    val=dict(
        raw_dir = '../datasets_summit/summit_sparse/val',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_sparse/val/val_features.pkl',
        # raw_dir='../datasets_summit/summit_sparse/val',
        # processed_dir='knn_processed',
        # features='../datasets_summit/summit_sparse/val/val_features.pkl',
        # raw_dir='../datasets_summit/summit/val',
        # processed_dir='knn_processed',
        # features='../datasets_summit/summit/val/val_features.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'ConstantAcceleration',
    obs_len=20,
    pred_len=30,
    avg_points=[20, 15, 10, 8, 6, 3],  # Number of predicted trajectories given past points. Must be at most history_points-1
    train_cfg = dict(

    ), # ConstantVelocity no needs training
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