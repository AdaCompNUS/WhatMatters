### ----------- Dataset settings --------------- ###
data_root_dir = '../datasets_summit/summit_3hz'
#data_root_dir = '../datasets_summit/summit_sparse' # for sparse data on Haoran's machine

data = dict(
    dataset_type = 'SummitDataset',
    train=dict(
        raw_dir=data_root_dir + '/train', # There must be a folder name 'data' inside this raw_dir
        processed_dir='knn_processed', # There no need absolute path. This is relative to raw_dir
        features='simulator/knearestneighbor/train_features.pkl',
        pipeline=[]
    ),
    test=dict(
        raw_dir = data_root_dir+'/val',
        processed_dir = 'knn_processed_testmode',
        features = data_root_dir+'/val/val_features_testmode.pkl',

        # raw_dir = '../datasets_summit/summit/val',
        # processed_dir='knn_processed_testmode',
        # features='../datasets_summit/summit/val/val_features_testmode.pkl',
        pipeline=[]
    ),
    val=dict(
        raw_dir=data_root_dir+'/val',
        processed_dir='knn_processed',
        features=data_root_dir+'/val/val_features.pkl',
        # raw_dir = '../datasets_summit/summit/val',
        # processed_dir='knn_processed',
        # features='../datasets_summit/summit/val/val_features.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'SummitKNearestNeighbor',
    normalize=True,
    use_delta=True,
    use_map=False,
    use_social=True,
    n_neigh=1,
    obs_len=20,
    pred_len=30,
    train_cfg = dict(
        save_model_path = "summit_knn_grid_search.pkl"
    ),
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/summit_knn_social_neigh1'

_base_ = [
    '../_base_/eval.py'
]