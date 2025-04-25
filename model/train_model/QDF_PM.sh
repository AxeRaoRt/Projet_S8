dataset="PM"

# Basis et grille utilisés dans le prétraitement.
basis_set=def2-SVP
radius=0.75
grid_interval=0.3
operation=sum
num_workers=0

# arguments à modifier si nous utilisons l'optimisation des hyperparamètres
batch_size=4
dim=200
hidden_HK=200
iteration=350
layer_HK=3
layer_functional=1
lr=0.006041
lr_decay=0.931770
step_size=18

predataset="SM"  # Extrapolation.

setting=$dataset--$basis_set--radius$radius--grid_interval$grid_interval--dim$dim--layer_functional$layer_functional--hidden_HK$hidden_HK--layer_HK$layer_HK--$operation--batch_size$batch_size--lr$lr--lr_decay$lr_decay--step_size$step_size--iteration$iteration
python QDF_PM.py $dataset $basis_set $radius $grid_interval $dim $layer_functional $hidden_HK $layer_HK $operation $batch_size $lr $lr_decay $step_size $iteration $setting $num_workers $predataset