dataset="SM"

# Basis set et grille utilisés dans le prétraitement.
basis_set=def2-SVP
radius=0.75
grid_interval=0.3

# Paramétrages de l'architecture du réseau de neurones.
dim=250  # On peut améliorer la performance en augmentant la dimension si hyperparam_optimizer est à "False"
layer_functional=4
hidden_HK=250
layer_HK=3

# Opération pour la couche finale.
operation=sum  # Opération d'agrégation de vecteurs utilisée dans le réseaux de neuronnes fonctionnal
# operation=mean  # For homo and lumo (i.e., a property unrelated to the molecular size or the unit is e.g., eV/atom).

# Setting of optimization.
batch_size=2
lr=1e-4
lr_decay=0.8
step_size=25
iteration=50

num_workers=0

hyperparam_optimizer="True"

setting=$dataset--$basis_set--radius$radius--grid_interval$grid_interval--dim$dim--layer_functional$layer_functional--hidden_HK$hidden_HK--layer_HK$layer_HK--$operation--batch_size$batch_size--lr$lr--lr_decay$lr_decay--step_size$step_size--iteration$iteration
python QDF_SM.py $dataset $basis_set $radius $grid_interval $dim $layer_functional $hidden_HK $layer_HK $operation $batch_size $lr $lr_decay $step_size $iteration $setting $num_workers $hyperparam_optimizer