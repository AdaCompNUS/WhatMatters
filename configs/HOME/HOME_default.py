### ----------- Dataset settings --------------- ###

data = dict(
    dataset_type = 'ArgoverseDataset',
    train=dict(
        features = 'model/HOME/internal/train',
    ),
    test=dict(
        features = 'model/HOME/internal/val',
    ),
    val=dict(
        features = 'model/HOME/internal/val',
    ),
)

### -----------  Model -----------  ###

model = dict(
    type = 'HOME',
    prepath = 'model/HOME/src/common_data_processing/script_vectorize_hd_maps.py',
    heatmappath = 'model/HOME/src/home/script_train_heatmap.py',
    predictorpath = 'model/HOME/src/home/script_train_trajectory_forecaster.py',
    valpath = 'model/HOME/src/home/script_evaluate_heatmap.py',
    config_path = 'configs/home.yaml',
)



### -------------- Evaluation ----------------- ###
work_dir = 'work_dirs/HOME_default/'

_base_ = [
    '../_base_/eval.py'
]