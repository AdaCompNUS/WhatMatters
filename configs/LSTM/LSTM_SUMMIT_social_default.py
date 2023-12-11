### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'SummitDataset',
    train=dict(
        raw_dir = '../datasets_summit/summit_10HZ/train',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_10HZ/train/train_features.pkl',
        pipeline = []
    ),
    test=dict(
        raw_dir = '../datasets_summit/summit_10HZ/val',
        processed_dir='knn_processed_testmode',
        features='../datasets_summit/summit_10HZ/val/val_features_testmode.pkl',
        pipeline=[]
    ),
    val=dict(
        raw_dir='../datasets_summit/summit_10HZ/val',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_10HZ/val/val_features.pkl',
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
        model_path = 'checkpoints/S_LSTM/summit',
        val_path = 'checkpoints/S_LSTM/summit/LSTM_rollout30.pth.tar',
    ),
    test_cfg = dict(
        test_batch_size = 512,
        joblib_batch_size = 100,
        model_path = 'checkpoints/S_LSTM/summit/LSTM_rollout30.pth.tar',
        traj_save_path = '../saved_trajectories_summit/lstm_social/rollout30_traj_test.pkl',
    ),
    val_cfg=dict(
        test_batch_size = 512,
        joblib_batch_size = 100,
        model_path = 'checkpoints/S_LSTM/summit/LSTM_rollout30.pth.tar',
        traj_save_path = '../saved_trajectories_summit/lstm_social/rollout30_traj_val.pkl',
        ## below for getting matrix ##
        metrics = True,
        gt = '../features/ground_truth_data/ground_truth_SummitDataset_10Hz_val.pkl',
        miss_threshold = 2,
        max_n_guesses = 6, # output k prediction modes
        prune_n_guesses = 0,
        n_guesses_cl = 0,
        n_cl = 0,
        max_neighbors_cl = 0,
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/LstmSocial_Summit/'

_base_ = [
    '../_base_/eval.py'
]