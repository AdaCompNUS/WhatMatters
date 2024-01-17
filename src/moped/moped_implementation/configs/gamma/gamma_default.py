### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    joblib_batch_size=100,
    train=dict(
        raw_dir = '/home/cunjun/moped_data/argoverse_gamma_samples/train/', # raw datafiles
        features='/home/cunjun/moped_data/argoverse_features/forecasting_features_train.pkl', # for evaluation
        pipeline = []
    ),
    test=dict(
        raw_dir='/home/cunjun/moped_data/argoverse_gamma_samples/test_obs/',
        features='/home/cunjun/moped_data/argoverse_features/forecasting_features_test.pkl',
        pipeline=[]
    ),
    val=dict(
        raw_dir= '/home/cunjun/moped_data/argoverse/val/',
        features='/home/cunjun/moped_data/argoverse_features/forecasting_features_val.pkl',
        pipeline=[]
    ),
    type_encode=dict(
        AGENT="Car",
        AV="CAR",
        OTHERS="People"
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'Gamma',
    obs_len=20,
    pred_len=30,
    k=6, # number of trajectories
    train_cfg = dict(

    ),
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/gamma_default/'

_base_ = [
    '../_base_/eval.py'
]