
dataset_trained=SM
basis_set=def2-SVP
radius=0.75
grid_interval=0.3
dataset_predict=data_test
property="False"


python preprocess_predict.py $dataset_trained $basis_set $radius $grid_interval $dataset_predict $property