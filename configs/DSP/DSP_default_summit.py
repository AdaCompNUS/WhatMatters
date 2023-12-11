### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'SummitDataset',
    joblib_batch_size=100,
    feature_dir = '../datasets_summit/summit_10HZ/features/',
    adv_cfg_path = 'model.DSP.config.dsp_cfg',
    train=dict(
        raw_dir='../datasets_summit/summit_10HZ/train',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_10HZ/train/train_features.pkl',
        pipeline=[]
    ),
    test=dict(
        raw_dir = '../datasets_summit/summit_10HZ/val',
        processed_dir='knn_processed_testmode',
        features='../datasets_summit/summit_10HZ/val/val_features_testmode.pkl',
        pipeline=[]
    ),
    val=dict(
        raw_dir = '../datasets_summit/summit_10HZ/val',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_10HZ/val/val_features.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'DSP',
    name = 'dsp',
    loss = 'dsp',
    obs_len=20,
    pred_len=30,
    use_cuda = True,
    model_path  = 'checkpoints/DSP/summit/20230324-021110_dsp_best.tar',
    k=6, # number of trajectories
    train_cfg = dict(
        clipping = 2.0,
        train_batch_size = 8,
        val_batch_size = 8,
        val_interval = 1,
        train_epoches = 50,
        logger_writer = False
    ),
    test_cfg = dict(
        shuffle = True
    ),
    val_cfg=dict(
        val_batch_size = 32
    )
)



### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/DSP_default_summit/'

_base_ = [
    '../_base_/eval.py'
]