
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Charger les données
data = pd.read_csv('output_SM.txt', sep='\t')


# Extraire les données pertinentes
epochs = data['Epoch']
mae_val = data['MAE_val(SM)']
mae_test = data['MAE_test(SM)']
loss_pce = data['Loss_PCE']  # Ajout de la colonne loss_pce


# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(epochs, mae_val, color='orange', label='validation')
plt.plot(epochs, loss_pce, color='blue', label='train')  # Ajout de la courbe loss_pce
#plt.plot(epochs, mae_test, color='green', label='Mae TEST') 
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Evolution du MAE en fonction des Epochs')

# Ajouter un cadre de texte avec les hyperparamètres
hyperparameters = """
dataset=SM
property=PCE
basis_set=def2-SVP
radius=0.75
grid_interval=0.3

dim=150
layer_functional=2
hidden_HK=200
layer_HK=2
operation=sum
batch_size=5
lr=0.020127
lr_decay=0.555957
step_size=2
iteration=177

"""
plt.text(0.95, 0.05, hyperparameters, horizontalalignment='right', verticalalignment='bottom', 
         transform=plt.gca().transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

plt.legend() # affiche la légende


# Afficher le graphique
plt.grid(True)
plt.tight_layout()
plt.show()

