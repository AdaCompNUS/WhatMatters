### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    joblib_batch_size=100,
    train=dict(
        features = '/home/data1/wuhr/argo_formal/forecasting_features_train.pkl',
        #features = 'data/argoverse_sample_features/forecasting_features_train.pkl',
        pipeline = []
    ),
    test=dict(
        features = '/home/data1/wuhr/argo_formal/forecasting_features_test.pkl',
        pipeline=[]
    ),
    val=dict(
        features = '/home/data1/wuhr/argo_formal/forecasting_features_val.pkl',
        pipeline=[]
    )
)

### -----------  Model -----------  ###

model = dict(
    type = 'KNearestNeighbor',
    normalize=False,
    use_delta=True,
    use_map=True,
    use_social=False,
    n_neigh=6,
    obs_len=20,
    pred_len=30,
    train_cfg = dict(

    ),
    test_cfg = dict(

    ),
    val_cfg=dict(

    )
)

### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/knearest_neighbor_map_nosocial_neigh6'

_base_ = [
    '../_base_/eval.py'
]