# echo "-- Processing val set..."
# python -W ignore data_argo/run_preprocess.py --mode val \
#   --data_dir /home/data1/wuhr/dataset/val/data/ \
#   --save_dir /home/data3/wuhr/data_argo/features/ 

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
 --data_dir /home/data1/wuhr/dataset/train/data/ \
 --save_dir /home/data3/wuhr/data_argo/features/

# echo "-- Processing test set..."
# python -W ignore data_argo/run_preprocess.py --mode test \
#   --data_dir /home/data1/wuhr/dataset/test_obs/data/ \
#   --save_dir /home/data3/wuhr/data_argo/features/ 
