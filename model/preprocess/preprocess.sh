dataset="SM"

# Basis set.
basis_set="def2-SVP" # ou basis_set=6-31G,dyall-ae3z etc selon le choix dans le site  : https://www.basissetexchange.org/  avec gto pout basis set

# Grid field.
radius=0.75
grid_inter=0.3

python preprocess.py $dataset $basis_set $radius $grid_inter