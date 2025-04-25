
datapath="../datasets/SM/SM.csv"
trainout="../datasets/SM/train/train_file"
valout="../datasets/SM/val/val_file"
testout="../datasets/SM/test/test_file"

python split_data.py  $datapath $trainout $valout $testout 

