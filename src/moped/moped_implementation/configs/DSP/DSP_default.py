### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    joblib_batch_size=100,
    feature_dir = '/home/data3/wuhr/data_argo/features/',
    adv_cfg_path = 'model.DSP.config.dsp_cfg',
    train=dict(
    ),
    test=dict(
    ),
    val=dict(
    ),
)

### -----------  Model -----------  ###

model = dict(
    type = 'DSP',
    name = 'dsp',
    loss = 'dsp',
    obs_len=20,
    pred_len=30,
    use_cuda = True,
    model_path  = 'model/DSP/saved_models/ckpt_dsp_epoch27.tar',
    k=6, # number of trajectories
    train_cfg = dict(
        clipping = 2.0,
        train_batch_size = 8,
        val_batch_size = 8,
        val_interval = 1,
        train_epoches = 30,
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
work_dir = 'work_dirs/DSP_default/'

_base_ = [
    '../_base_/eval.py'
]