import argparse
from collections import defaultdict
import os
import pickle

import numpy as np

from scipy import spatial

import basis_set_exchange as bse
from mendeleev import element

# dictionnaire de nombres atomiques -> symboles

periodic_table_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
                'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

atomic_numbers_dict = {periodic_table_atoms[i-1] : i for i in range(1, len(periodic_table_atoms)+1)}

def create_sphere(radius, grid_interval):  # on créee une sphere pour modeliser chaque atome de la molécule
    xyz = np.arange(-radius, radius+1e-3, grid_interval)
    sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz  # len(xyz) * len(xyz) * len(xyz) possibilité de points
              if (x**2 + y**2 + z**2 <= radius**2) and [x, y, z] != [0, 0, 0]]  # si le point est dans la sphère et différent de l'atome
    return np.array(sphere)

def create_field(sphere, coords):  # a chaue atome on attribue un champ
    field = [f for coord in coords for f in sphere+coord]  # on superpose les positions de l'atome et du champ
    return np.array(field)

def create_distancematrix(coords1, coords2):
    distance_matrix = spatial.distance_matrix(coords1, coords2)  # créée une matrice de distance entre les points du champ et les atomes
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)  # remplace les 0 par 1e6

def create_orbitals(orbitals, orbital_dict):
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)

def create_potential(distance_matrix, atomic_numbers):
    Gaussians = np.exp(-distance_matrix**2)  
    return -1  * np.matmul(Gaussians, atomic_numbers)  # Le résultat matriciel est multiplié par -1 pour signifier un potentiel négatif

def create_dataset(dir_dataset, filename, basis_set, radius, grid_interval, orbital_dict, property=True):
    
    # Définit le répertoire de sortie pour les données prétraitées
    if property:
        dir_preprocessed = (dir_dataset + filename + "/" + filename + '_' + basis_set + "_sphere_" + str(radius) + "_" + str(grid_interval) + 'grid/')
    else:  
        dir_preprocessed = dir_dataset + filename + '/' + filename + '_' + basis_set
    
    # Crée le répertoire de sortie s'il n'existe pas
    os.makedirs(dir_preprocessed, exist_ok=True)
    
    # Récupère les métadonnées du basis set depuis Basis Set Exchange
    try:
        metadata = bse.filter_basis_sets()[basis_set.lower()]
        latest_version = metadata.get("latest_version")
        last_element_in_basis_set = len(metadata['versions'][str(latest_version)]['elements'])
    except KeyError:
         raise ValueError(f"Basis set '{basis_set}' non trouvé dans Basis Set Exchange.")
    
    # Vérifie que le basis set utilise des orbitales de type Gaussien (GTO)
    if not "gto" in metadata['function_types']:
        raise ValueError("Le basis set fourni n'utilise pas de fonctions GTO.")

    # Télécharge les informations détaillées du basis set pour les éléments concernés
    basis_set_exchange = bse.get_basis(basis_set, elements=[i for i in range(1, last_element_in_basis_set + 1)])
    
    if not basis_set_exchange:
        raise ValueError(f"Invalid basis set: {basis_set}")
    
    # Crée la sphère de points une seule fois
    sphere = create_sphere(radius, grid_interval)
    
    # On récupère les fichiers avec les coordonnées
    with open(dir_dataset + filename + "/" + filename + "_m3d" + '.txt', 'r') as f:
        dataset = f.read().strip().split('\n\n')
    
    # Traite chaque molécule dans le jeu de données
    for n, data in enumerate(dataset):
        data = data.strip().split('\n')
        
        # Extrait l'identifiant de la molécule
        idx = data[0]
        print(f"Traitement de la molécule : {idx}") # Ajout pour suivi
        
        # Extrait les coordonnées atomiques et éventuellement les propriétés
        if property:
            if len(data) < 3:
                print(f"Attention : Données incomplètes pour {idx}. Skipping.")
                continue
            atom_xyzs = data[1:-1]
            property_values_str = data[-1].strip().split()
            property_values = np.array([[float(p) for p in property_values_str]])
            
        else:
            if len(data) < 2:
                print(f"Attention : Données incomplètes pour {idx}. Skipping.")
                continue
            atom_xyzs = data[1:]
            
        # Initialise les listes pour stocker les informations de la molécule
        atoms = []
        atomic_numbers = []
        N_electrons = 0
        atomic_coords = []
        atomic_orbitals = []
        orbital_coords = [] # Coordonnées associées à chaque orbitale (identiques aux coords atomiques)
        quantum_numbers = [] # Nombre quantique principal associé à chaque orbitale
        
        # Traite chaque ligne atome/coordonnée
        for atom_xyz in atom_xyzs:
            atom, x, y, z = atom_xyz.split()
            atoms.append(atom)
            
            atomic_number = atomic_numbers_dict[atom]
            atomic_numbers.append([atomic_number])
            
            N_electrons += atomic_number
            
            xyz = [float(v) for v in [x, y, z]]
            atomic_coords.append(xyz)
            
            electronic_configuration = element(atom).ec.conf  # on recupere la configuration électronique de l'atome actuel
            
            aqs = [] # Liste pour stocker (nom_orbitale, nombre_quantique_principal)
            
            number_of_functions_primitive = 0
            try:   # si un element n'est pas trouvé dans le basis set, on l'ignore
                electron_shell = basis_set_exchange["elements"][str(atomic_number)]['electron_shells']
            except KeyError:
                print(f"Warning: Numéro atomique de l'élément {atomic_number} n'est pas trouvé dans le basis set. Skipping element {atom}.")
                continue
            
            for atomic_basis_function in  electron_shell:
                
                i = electron_shell.index(atomic_basis_function)
                    
                if i < len(electron_shell) - 1:
                    if electron_shell[i]["angular_momentum"] != electron_shell[i-1]["angular_momentum"]:
                        number_of_functions_primitive = 0
                
                if atomic_basis_function["angular_momentum"] == [0]:
                    for orbital in electronic_configuration:
                        if orbital[1] == "s":
                            for i in range(len(atomic_basis_function["exponents"])):  # on ajoute les orbitales s
                                aqs.append((atom + str(orbital[0]) + orbital[1] + str(number_of_functions_primitive + i), orbital[0]))  # orbital[0] = nombre quantique principal
                    number_of_functions_primitive += len(atomic_basis_function["exponents"])
                
                elif atomic_basis_function["angular_momentum"] == [1]:
                    for orbital in electronic_configuration:
                        if orbital[1] == "p":
                            for i in range(len(atomic_basis_function["exponents"])):  # on ajoute les orbitales p
                                aqs.append((atom + str(orbital[0]) + orbital[1] + str(number_of_functions_primitive + i), orbital[0]))  # orbital[0] = nombre quantique principal
                    number_of_functions_primitive += len(atomic_basis_function["exponents"])
                
                elif atomic_basis_function["angular_momentum"] == [2]:
                    for orbital in electronic_configuration:
                        if orbital[1] == "d":
                            for i in range(len(atomic_basis_function["exponents"])):  # on ajoute les orbitales d
                                aqs.append((atom + str(orbital[0]) + orbital[1] + str(number_of_functions_primitive + i), orbital[0]))  # orbital[0] = nombre quantique principal
                    number_of_functions_primitive += len(atomic_basis_function["exponents"])
                
                elif atomic_basis_function["angular_momentum"] == [3]:
                    for orbital in electronic_configuration:
                        if orbital[1] == "f":
                            for i in range(len(atomic_basis_function["exponents"])):  # on ajoute les orbitales f
                                aqs.append((atom + str(orbital[0]) + orbital[1] + str(number_of_functions_primitive + i), orbital[0]))  # orbital[0] = nombre quantique principal
                    number_of_functions_primitive += len(atomic_basis_function["exponents"])
            
            # Ajoute les orbitales et leurs informations associées pour cet atome
            for orbital_name, n_quantum in aqs:
                atomic_orbitals.append(orbital_name)
                orbital_coords.append(xyz) # Chaque orbitale est centrée sur l'atome
                quantum_numbers.append(n_quantum)
        
        # Vérifie si des atomes/orbitales valides ont été traités pour cette molécule
        if not atomic_coords or not atomic_orbitals:
             print(f"Attention : Aucune donnée atomique/orbitale valide traitée pour {idx}. Skipping molecule.")
             continue
             
        # Convertit les listes en tableaux NumPy
        atomic_coords = np.array(atomic_coords)
        atomic_orbitals_encoded = create_orbitals(atomic_orbitals, orbital_dict) # Encode les noms d'orbitales en entiers
        orbital_coords = np.array(orbital_coords)
        quantum_numbers = np.array([quantum_numbers]) # Ajoute une dimension pour correspondre au format attendu
        atomic_numbers = np.array(atomic_numbers)
        N_electrons = np.array([[N_electrons]])
        
        # Crée le champ de points autour de la molécule
        field_coords = create_field(sphere, atomic_coords)
        N_field = len(field_coords)
        
        # Calcule la matrice des distances entre le champ et les atomes
        distance_matrix_pot = create_distancematrix(field_coords, atomic_coords)
        # Calcule le potentiel externe
        potential = create_potential(distance_matrix_pot, atomic_numbers)
        
        # Calcule la matrice des distances entre le champ et les centres des orbitales
        distance_matrix_orb = create_distancematrix(field_coords, orbital_coords)
        
        # Structure les données prétraitées pour la sauvegarde
        data_to_save = [idx,
                        atomic_orbitals_encoded.astype(np.int64), # Indices des orbitales
                        distance_matrix_orb.astype(np.float32), # Distances champ <-> orbitales
                        quantum_numbers.astype(np.float32), # Nombres quantiques principaux
                        orbital_coords.astype(np.float32), # Coordonnées des centres des orbitales
                        N_electrons.astype(np.float32), # Nombre total d'électrons
                        N_field, # Nombre de points dans le champ
                       ]
        
        # Ajoute les propriétés et le potentiel si disponibles
        if property:
            data_to_save.append(property_values.astype(np.float32)) 
            data_to_save.append(potential.astype(np.float32)) 
        else:
             data_to_save.append(np.array([[]], dtype=np.float32)) 
             data_to_save.append(potential.astype(np.float32)) 
             
        # Convertit la liste en un tableau NumPy de type 'object' car les éléments ont des formes différentes
        data_array = np.array(data_to_save, dtype=object)
        
        # Sauvegarde les données prétraitées pour cette molécule
        output_path = os.path.join(dir_preprocessed, idx + ".npy")
        np.save(output_path, data_array)
        
    print(f"Prétraitement terminé. Données sauvegardées dans : {dir_preprocessed}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_inter', type=float)
    args = parser.parse_args()
    dataset = args.dataset
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_inter
    
    dir_dataset = '../../datasets/' + dataset + '/'
    
    """ Initialise le dictionnaire orbital_dict, dans lequel
    chaque clé est un type d'orbitale et chaque valeur est son index."""
    
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    print('Preprocess', dataset, 'dataset.\n'
        'Le preprocessed dataset est enregistré dans le répertoire', dir_dataset, 'répertoire.\n'
        'Si la taille du dataset est grande, '
        'cela prend beaucoup de temps et consomme du stockage.\n'
        'Attendez un moment...')
    
    print('-'*50)
    
    print('Training dataset...')
    create_dataset(dir_dataset, 'train', basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)

    print('Validation dataset...')
    create_dataset(dir_dataset, 'val',
                   basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)

    print('Test dataset...')
    create_dataset(dir_dataset, 'test', basis_set, radius, grid_interval, orbital_dict)
    print('-'*50)
    
    os.makedirs(dir_dataset + 'orbitaldict_' + dataset + "/", exist_ok=True)
    
    with open(dir_dataset + 'orbitaldict_' + dataset + "/" + 'orbitaldict_' + basis_set + '.pickle', 'wb') as f:
        pickle.dump(dict(orbital_dict), f)

    print('Le preprocessing du dataset est terminé.\n')