# moped

### 1. Summary of supported models （22.10.11）
The current motion prediction supports:

|        Model        | Train | Test | Validation | 
|:-------------------:|:-----:|:----:|:----------:|
| ConstantVelocity    |  Yes  | Yes  |    Yes     |
| ConstantAceleration |  Yes  | Yes  |    Yes     |
|  KNearestNeighbor   |  Yes  | Yes  |    Yes     |
|        Gamma        |  No   | Yes  |    Yes     |
|        LSTM         |  Yes  |  No  |    Yes     |
|       S-LSTM        |  Yes  |  No  |    Yes     |
|   Map_prior-LSTM    |  Yes  |  No  |    Yes     |
|        HiVT         |  Yes  | Yes  |    Yes     |
|       LaneGCN       |  Yes  | Yes  |    Yes     |
|         DSP         |  Yes  | Yes  |    Yes     |
|    mmTransformer    |  No   |  No  |    Yes     |
|    Trajectron++     |  No   |  No  |     No     |
|        Jean         |  No   |  No  |     No     |

### 2. How to run each model?

#### 2.1 ConstantVelocity
This is a simple model that predict the next position by adding averaged 
velocity from previous n history points.

To generate competition files, run:
```
python test.py configs/argoverse_baseline/constant_velocity_n30_p6.py --test
```

To validate the prediction model, run:
```
python test.py configs/argoverse_baseline/constant_velocity_n30_p6.py
```
The validation configurations are default according to argoverse-forecasting
repository. Please update if you want to run with different configs 
when validating

#### 2.2 LSTM_based
We add Gaussian noises to the decoder to make the LSTM-based method a multi-modal method.

Besides, we treat the future trajectory of val set as unknown, and generate the candidate centerline for it when using map-prior lstm method. Just run:
```
python compute_features.py --data_dir ../datasets/val/data --feature_dir ../features/forecasting_features --mode test --name val_testmode --obs_len 20 --pred_len 30
```

To train/validate the prediction model, run:
```
python train.py/test.py configs/LSTM/LSTM_basic_default.py

python train.py/test.py configs/LSTM/LSTM_social_default.py

python train.py/test.py configs/LSTM/LSTM_map_default.py
```

#### 2.3 LaneGCN & DSP
Similar to methods above, to train/validate the prediction model, run:
```
python train.py/test.py configs/LaneGCN/LaneGCN_default.py
python train.py/test.py configs/DSP/DSP_default.py
```
To generate competition files, run:
```
python test.py configs/LaneGCN/LaneGCN_default.py --test
python test.py configs/DSP/DSP_default.py --test
```

#### 2.4 mmtransformer--Only Validation
```
python Evaluation.py ./config/demo.py --model-name demo
```


### 3. Results Summary
#### 3.1 On Argoverse Validation Dataset
|                         Model                          | brier-minFDE(K=6) | minFDE(K=6) | MR(K=6) | minADE(K=6) | DAC(K=6) | minFDE(K=1) | MR(K=1) | minADE(K=1) |
|:------------------------------------------------------:|:-----------------:|:-----------:|:-------:|:-----------:|:--------:|:-----------:|:-------:|:-----------:|
|     ConstantVelocity (constant_velocity_n30_p1.py)     |        4.591        |    3.897    |   0.559 |     1.727     |  0.9149  |   6.0534    | 0.7421  |   2.7151    |
| ConstantAcceleration (constant_acceleration_n30_p1.py) |        4.705        |     4.011    |   0.555   |     1.773     |  0.9118  |   6.174    | 0.7482  |    2.777    |
|      KNearestNeighbor (knn_nomap_nosocial_neigh6)      |      3.2738       |   2.5793    | 0.4284  |    1.412    |  0.9004  |    6.721    |  0.816  |    2.99     |
|                         Gamma                          |     11.93798      |   11.2435   | 0.90233 |   5.4652    |  0.7576  |   14.5556   | 0.9643  |   7.0107    |
|                          lstm                          |       4.554       |    3.860    |  0.643  |    1.694    |  0.955   |    3.948    | 0.6578  |    1.728    |
|                         s-lstm                         |       4.512       |    3.818    |  0.634  |    1.688    |  0.952   |    3.954    |  0.656  |    1.741    |
|                     mapPrior-lstm                      |       4.608       |    3.914    |  0.627  |    1.911    |   N.A.   |    5.966    |  0.700  |    2.660    |
|                          HiVT                          |       1.663       |    0.969    |  0.092  |   0.661     |   N.A.   |    3.547    |  0.599  |    1.609    |
|                        LaneGCN                         |       1.775       |    1.081    |  0.103  |   0.7115    |   N.A.   |    3.020    |  0.499  |    1.367    |
|                          DSP                           |       1.647       |    1.015    |  0.096  |    0.727    |   N.A.   |    3.049    |  0.515  |    1.415    |
|                     mmtransformer                      |       1.951       |    1.081    |  0.102  |   0.7098    |  0.9902  |    5.595    |  0.836  |    2.298    |

 
#### 3.2 On Argoverse Test Dataset
|                         Model                          | brier-minFDE(K=6) | minFDE(K=6) | MR(K=6) | minADE(K=6) | DAC(K=6) | minFDE(K=1) | MR(K=1)  | minADE(K=1) |
|:------------------------------------------------------:|:-----------------:|:-----------:|:-------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|
|    ConstantVelocity  (constant_velocity_n30_p1.py)     |        N.A        |     N.A     |   N.A   |     N.A     |  0.885   |     7.887   | 0.834    |   3.53      |
| ConstantAcceleration (constant_acceleration_n30_p1.py) |        N.A        |     N.A     |   N.A   |     N.A     | 0.8846   |    8.0324   | 0.83838  |   3.6093    |
|    KNearestNeighbor  (knn_nomap_nosocial_neigh6.py)    |      3.9908       |   3.2963    | 0.9228  |   1.7175    |  0.8672  |    7.911    |  0.871   |    3.468    |
|               Gamma   (gamma_default.py)               |      11.6254      |   11.625    |  0.947  |    5.686    |  0.8673  |   11.625    |  0.947   |    5.686    |
|                          HiVT                          |                   |             |         |             |          |             |          |             |
|                        LaneGCN                         |       2.049       |    1.355    | 0.1597  |   0.8679    |  0.9835  |    3.764    |  0.587   |   1.7023    |
|                          DSP                           |      1.9163       |   1.2634    | 0.1436  |   0.8608    |  0.9899  |   3.8986    |  0.6022  |   1.7937    |

#### 3.3 On SUMMIT Validation Dataset 

|                             Model                       | brier-minFDE(K=6) | minFDE(K=6) | MR(K=6) | minADE(K=6) | DAC(K=6) | minFDE(K=1) | MR(K=1)  | minADE(K=1) |
|:-------------------------------------------------------:|:-----------------:|:-----------:|:-------:|:-----------:|:--------:|:-----------:|:--------:|:-----------:|
|  ConstantVelocity  (summit_constant_velocity_n30_p6.py) |      4.4322       |   3.7377    |  0.443  |   1.8669    |   N.A.   |   6.6930    | 0.73894  |   3.1474    |
|ConstantAcceleration(summit_constant_acceleration_n30_p6.py)|   4.5759       |    3.881    | 0.4425  |   1.94669   |   N.A.   |   7.14454   | 0.73686  |   3.40218   |
|     KNN  (summit_knn_map_nosocial_delta_neigh6.py)      |      3.2197       |   2.5253    | 0.37349 |   2.1606    |   N.A.   |  10.76683   |  0.6910  |   5.4749    |
|        KNN  (summit_knn_map_nosocial_neigh6.py)         |      3.9056       |   3.2111    | 0.42104 |   3.0717    |   N.A.   |  10.54548   |  0.7141  |   6.38845   |
|    KNN  (summit_knn_nomap_nosocial_delta_neigh6.py)     |      3.4366       |   2.74217   | 0.395   |  1.6678     |   N.A.   |  6.5054     | 0.6541   |    3.0997   |
|        LSTM  (LSTM_SUMMIT_basic_default.py)             |      5.5649       |   4.8705    |  0.6529 |  2.3117     |   N.A.   |  5.0055     | 0.6647   |    2.3718   |
|        LSTM  (LSTM_SUMMIT_social_default.py)            |      5.6003       |   4.9059    | 0.6762  |  2.3685     |   N.A.   |  5.1360     |  0.7067  |    2.4687   |
|        LSTM  (LSTM_SUMMIT_map_default.py)               |      5.6514       |   4.9570    |  0.6578 |  2.8771     |   N.A.   |  6.0375     |  0.6798  |    3.4001   |
|        LaneGCN  (LaneGCN_summit_default.py)             |      2.1891       |   1.4947    |  0.2334 |  1.0094     |   N.A.   |  4.2149     |  0.4749  |    1.9536   |
|        HiVT  (hivt_summit_default.py)                   |      1.7592       |   1.0648    |  0.1635 |  0.8077     |   N.A.   |  3.5246     |  0.4619  |    1.6920   |

### 4. Motion Planning Results

|                                      Model             | Collision rate | Average speed | Col rate per meter | Col rate per step | Travelled dist avg | Travelled dist total | Total run files |
|:------------------------------------------------------:|:--------------:|:-------------:|:------------------:|:-----------------:|:------------------:|:--------------------:|:---------------:|
|     ConstantVelocity (constant_velocity_n30_p6.py)     |      0.15      |     2.345     |       0.001        |      0.0014       |       139.26       |        13926         |       100       |
| ConstantAcceleration (constant_acceleration_n30_p6.py) |     0.1165     |     2.095     |       0.0009       |       0.001       |       128.55       |        13276         |       103       |
