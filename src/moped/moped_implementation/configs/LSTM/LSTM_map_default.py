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
    type = 'LSTM',
    obs_len=20,
    pred_len=30,
    normalize = False,
    use_delta = True,
    use_map  = True,
    use_social  = False,
    lr = 0.001,
    train_cfg = dict(
        end_epoch = 5000,
        train_batch_size = 512,
        val_batch_size = 512,
        model_path = 'model/LSTM/saved_models',
    ),
    test_cfg = dict(
        
    ),
    val_cfg=dict(
        test_batch_size = 512,
        joblib_batch_size = 100,
        model_path = 'model/LSTM/saved_models/lstm_map/LSTM_rollout30.pth.tar',
        traj_save_path = '../saved_trajectories/lstm_map/rollout30_traj_val.pkl',
        ## below for getting matrix ##
        metrics = True,
        gt = '../features/ground_truth_data/ground_truth_ArgoverseDataset_val.pkl',
        miss_threshold = 2,
        max_n_guesses = 0,
        prune_n_guesses = 0,
        n_guesses_cl = 1,
        n_cl = 6,
        max_neighbors_cl = 3,
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LstmMap/'

_base_ = [
    '../_base_/eval.py'
]