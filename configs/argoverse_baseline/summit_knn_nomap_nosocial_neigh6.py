### ----------- Dataset settings --------------- ###
data = dict(
    dataset_type = 'SummitDataset',
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
        raw_dir='../datasets_summit/summit_10HZ/val',
        processed_dir='knn_processed',
        features='../datasets_summit/summit_10HZ/val/val_features.pkl',
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
    n_neigh=6,
    obs_len=20,
    pred_len=30,
    checkpoint = "checkpoints/KNN/summit_knn_nomap_nosocial.pkl",
    train_cfg = dict(
        save_model_path = "summit_knn_nomap_nosocial.pkl"
    ),
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/summit_knn_nomap_nosocial_neigh6'

_base_ = [
    '../_base_/eval.py'
]