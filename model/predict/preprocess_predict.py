import argparse
from collections import defaultdict
import glob
import pickle
import shutil
import sys

sys.path.append('../')
from preprocess import preprocess as pp

def load_dict(filename):
    with open(filename, 'rb') as f:
        dict_load = pickle.load(f)
        dict_default = defaultdict(lambda: max(dict_load.values())+1)
        for k, v in dict_load.items():
            dict_default[k] = v
    return dict_default

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_interval', type=float)
    parser.add_argument('dataset_predict')
    parser.add_argument('property')
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dataset_predict = args.dataset_predict
    property = args.property

    dir_trained = '../../datasets/' + dataset_trained + '/'
    dir_predict = '../../datasets/' + dataset_predict + '/'

    filename = dir_trained + 'orbitaldict_' + dataset_trained + "/" + 'orbitaldict_' + basis_set + '.pickle'
    orbital_dict = load_dict(filename)
    N_orbitals = len(orbital_dict)
    
    if property == 'True':
        property = True
    else:
        property = False
    
    print('Le preprocess', dataset_predict, 'dataset.\n'
          'Le jeu de données prétraité est enregistré dans le répertoire', dir_predict, '\n'
          'Si la taille du jeu de données est grande, '
          'cela prend beaucoup de temps et consomme du stockage.\n'
          'Attendez un moment...')
    print('-'*50)
    
    print("property = ", property)
    pp.create_dataset(dir_predict, 'demo',
                    basis_set, radius, grid_interval, orbital_dict, property=property)
    
    filename = dir_trained + 'orbitaldict_' + dataset_trained + "/" + 'orbitaldict_' + basis_set + '.pickle'
    orbital_dict = load_dict(filename)

    if N_orbitals < len(orbital_dict):
        print('##################### Warning!!!!!! #####################\n'
            'Le dataset de prédiction peut contenir des atomes inconnus\n'
            'qui n\'ont pas encore été appris dans le jeu de données d\'entraînement.\n'
            'Les paramètres de ces atomes n\'ont pas encore été appris\n'
            'et doivent être initialisés aléatoirement cette fois-ci.\n')
        
    print('-'*50)
    print('Entrainement terminé.')