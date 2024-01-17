### ----------- Dataset settings --------------- ###
data_root_dir = '../datasets_summit/summit_3hz'
#data_root_dir = '../datasets_summit/summit_sparse' # for sparse data on Haoran's machine

data = dict(
    dataset_type = 'SummitDataset',
    train=dict(
        raw_dir=data_root_dir + '/train',
        processed_dir='knn_processed',
        features='simulator/knearestneighbor/train_features.pkl',
        pipeline=[]
    ),
    test=dict(
        raw_dir = data_root_dir+'/val',
        processed_dir = 'knn_processed',
        features = data_root_dir+'/val/val_features.pkl',

        pipeline=[]
    ),
    val=dict(
        raw_dir=data_root_dir+'/val',
        processed_dir='knn_processed',
        features=data_root_dir+'/val/val_features.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'SummitKNearestNeighbor',
    normalize=True,
    use_delta=True,
    use_map=False,
    use_social=False,
    n_neigh=1,
    obs_len=20,
    pred_len=30,
    train_cfg = dict(
        save_model_path = "summit_knn_grid_search.pkl"
    ),
    test_cfg = dict(

    ),
    val_cfg=dict(
        save_model_path = "simulator/knearestneighbor/knn/summit_knn_grid_search.pkl"
    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/summit_knn_default_neigh1'

_base_ = [
    '../_base_/eval.py'
]