### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'SummitDataset',
    joblib_batch_size=100,
    root_dir= '../datasets_summit/summit_10HZ',
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
    type = 'HiVT',
    obs_len=20,
    pred_len=30,
    k=6,
    train_cfg = dict(
        embed_dim = 128,
        gpus = 4,
        train_batch_size = 8,
    ),
    test_cfg = dict(

    ),
    val_cfg=dict(
        val_batch_size = 32,
        ckpt_path = 'checkpoints/HiVT/summit/epoch=63-step=95935.ckpt',
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/hivt_default/' # the last dirname should be same as this filename for easy to track

_base_ = [
    '../_base_/eval.py'
]