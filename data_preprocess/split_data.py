import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
import argparse

 
def generate_dataset(datapath, trainout, valout, testout):
    
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(datapath):
        raise FileNotFoundError(f"Le fichier {datapath} n'existe pas.")
   
    # Lire le fichier CSV
    print(f"Lecture du fichier : {datapath}")
    data = pd.read_csv(datapath)
    print("Fichier lu avec succès.")
    print(f"Nombre de lignes : {len(data)}")
   
    # Arrondir les valeurs de PCE
    data["PCE"] = np.floor(data["PCE"])
   
    # Définir les caractéristiques (X) et la cible (y)
    X = data['SMILES']
    y = data['PCE']
   
   
    # Mettre à jour X et y
    X = data['SMILES']
    y = data['PCE']
   
    # Créer les répertoires si ils n'existent pas
    os.makedirs(os.path.dirname(trainout), exist_ok=True)
    os.makedirs(os.path.dirname(valout), exist_ok=True)
    os.makedirs(os.path.dirname(testout), exist_ok=True)
   
    
    
    
    # Diviser les données en ensembles d'entraînement et de test
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    
    
    for train_index, test_index in ss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Sauvegarder l'ensemble de test
        test = data.iloc[test_index]
        try:
            print(f"Sauvegarde du fichier test dans : {testout}.csv")
            test.to_csv(testout + '.csv', index=False)
            test.to_csv(testout + '.txt', index=False)
            print("Fichier test sauvegardé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier test : {e}")
       
        # # Diviser l'ensemble d'entraînement en entraînement et validation
        
    ss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=21)
    for train_idx, val_idx in ss_val.split(X_train, y_train):
        # Sauvegarder l'ensemble d'entraînement
        train = data.iloc[train_index].iloc[train_idx]
        try:
            print(f"Sauvegarde du fichier train dans : {trainout}.csv")
            train.to_csv(trainout + '.csv', index=False)
            train.to_csv(trainout + '.txt', index=False)
            print("Fichier train sauvegardé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier train : {e}")
        
        # Sauvegarder l'ensemble de validation
        val = data.iloc[train_index].iloc[val_idx]
        try:
            print(f"Sauvegarde du fichier val dans : {valout}.csv")
            val.to_csv(valout + '.csv', index=False)
            val.to_csv(valout + '.txt', index=False)
            print("Fichier val sauvegardé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier val : {e}")
            
   
    return print('Les datasets ont été générés avec succès !')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='Path to the input CSV file')
    parser.add_argument('trainout', type=str, help='Path to save the training dataset')
    parser.add_argument('valout', type=str, help='Path to save the validation dataset')
    parser.add_argument('testout', type=str, help='Path to save the test dataset')
    
    args = parser.parse_args()
    
    datapath = args.datapath
    trainout = args.trainout
    valout = args.valout
    testout = args.testout
    
    generate_dataset(datapath, trainout, valout, testout)