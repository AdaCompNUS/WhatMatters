echo "-- Processing val set..."
python -W ignore data_argo/run_preprocess.py --mode val \
  --data_dir /home/cunjun/wuhr/datasets/val/data/ \
  --save_dir /home/cunjun/wuhr/datasets/features/ \
  --dataset Argoverse

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
 --data_dir /home/cunjun/wuhr/datasets/train/data/ \
 --save_dir /home/cunjun/wuhr/datasets/features/  \
 --dataset Argoverse
