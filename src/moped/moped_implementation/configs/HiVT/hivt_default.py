### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    joblib_batch_size=100,
    root_dir= '../datasets/',
    train=dict(
        features = '../features/forecasting_features/forecasting_features_train.pkl',
        pipeline = []
    ),
    test=dict(
        features = '../features/forecasting_features/forecasting_features_test.pkl',
        pipeline=[]
    ),
    val=dict(
        features = '../features/forecasting_features/forecasting_features_val.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'HiVT',
    obs_len=20,
    pred_len=30,
    k=6, # number of trajectories
    train_cfg = dict(
        embed_dim = 64,
        gpus = 4,
        train_batch_size = 8,
    ),
    test_cfg = dict(

    ),
    val_cfg=dict(
        val_batch_size = 32,
        # ckpt_path = 'model/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt',
        ckpt_path = 'model/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt',
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/hivt_default/' # the last dirname should be same as this filename for easy to track

_base_ = [
    '../_base_/eval.py'
]