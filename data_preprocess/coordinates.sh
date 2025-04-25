
file_path="../datasets/SM/train/train_file.csv"
output_file="../datasets/SM/train/train_m3d.txt"

python coordinates.py $file_path $output_file

file_path="../datasets/SM/val/val_file.csv"
output_file="../datasets/SM/val/val_m3d.txt"

python coordinates.py $file_path $output_file

file_path="../datasets/SM/test/test_file.csv"
output_file="../datasets/SM/test/test_m3d.txt"

python coordinates.py $file_path $output_file