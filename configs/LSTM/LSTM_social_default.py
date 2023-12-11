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
    normalize = True,
    use_delta = True,
    use_map  = False,
    use_social  = True,
    lr = 0.001,
    train_cfg = dict(
        end_epoch = 5000,
        train_batch_size = 512,
        val_batch_size = 512,
        model_path = 'checkpoints/S_LSTM/argo',
        val_path = 'checkpoints/S_LSTM/argo/LSTM_rollout30.pth.tar',
    ),
    test_cfg = dict(
        test_batch_size = 512,
        joblib_batch_size = 100,
        model_path = 'checkpoints/S_LSTM/argo/LSTM_rollout30.pth.tar',
        traj_save_path = '../saved_trajectories/lstm_social/rollout30_traj_test.pkl',
    ),
    val_cfg=dict(
        test_batch_size = 512,
        joblib_batch_size = 100,
        model_path = 'checkpoints/S_LSTM/argo/LSTM_rollout30.pth.tar',
        traj_save_path = '../saved_trajectories/lstm_social/rollout30_traj_val.pkl',
        ## below for getting matrix ##
        metrics = True,
        gt = '../features/ground_truth_data/ground_truth_ArgoverseDataset_val.pkl',
        miss_threshold = 2,
        max_n_guesses = 6, # output k prediction modes
        prune_n_guesses = 0,
        n_guesses_cl = 0,
        n_cl = 0,
        max_neighbors_cl = 0,
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LstmSocial/'

_base_ = [
    '../_base_/eval.py'
]