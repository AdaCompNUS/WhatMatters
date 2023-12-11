echo "-- Processing summit val set..."
python -W ignore data_argo/run_preprocess.py --mode val \
  --data_dir /home/cunjun/wuhr/datasets_summit/summit_10HZ/val/data/ \
  --save_dir /home/cunjun/wuhr/datasets_summit/summit_10HZ/features/ \
  --dataset summit

echo "-- Processing summit train set..."
python data_argo/run_preprocess.py --mode train \
 --data_dir /home/cunjun/wuhr/datasets_summit/summit_10HZ/train/data/ \
 --save_dir /home/cunjun/wuhr/datasets_summit/summit_10HZ/features/ \
 --dataset summit
