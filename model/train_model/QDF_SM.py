
import argparse
import os
from pathlib import Path
import pickle
import timeit

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from e3nn.o3 import spherical_harmonics
import torch.utils.data
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

os.environ["CUDA_LAUNCH_BLOCKING"] = "1" # pour deboguer les erreurs CUDA

class QuantumDeepField(nn.Module):
    
    def __init__(self, device, N_orbitals, dim, layer_functional, operation, N_output, hidden_HK, layer_HK):
        super(QuantumDeepField, self).__init__()    
        
        self.coefficient = nn.Embedding(N_orbitals, dim) # embedings de chaque orbitale avec dim comme dimension
        self.zeta = nn.Embedding(N_orbitals, 1) 
        nn.init.ones_(self.zeta.weight) 
        
        self.W_functional = nn.ModuleList([nn.Linear(dim, dim) for i in range(layer_functional)]) # layer pour prédire les propriétés
        self.bn_functional = nn.ModuleList([nn.BatchNorm1d(dim) for i in range(layer_functional)])
        self.dropout_functional = nn.ModuleList([nn.Dropout(p=0.15) for i in range(layer_functional)])

        self.W_property = nn.Linear(dim, N_output)
        
        
        self.W_density = nn.Linear(1, hidden_HK) # layer pour les contraintes de HK ayant comme entrée la densité
        self.W_HK = nn.ModuleList([nn.Linear(hidden_HK, hidden_HK) for i in range(layer_HK)])
        self.W_potential = nn.Linear(hidden_HK, 1)
        
        self.device = device # device = 'cuda' or 'cpu'
        self.dim = dim
        self.layer_functional = layer_functional
        self.operation = operation  # operation = 'sum' or 'concatenate'
        self.layer_HK = layer_HK
        
        self.prelu = nn.PReLU()
    
    def list_to_batch(self, xs, dtype=torch.FloatTensor, cat=None, axis=None):
        """Transforme une liste de données numpy en un batch de tenseurs PyTorch."""
        xs = [dtype(x).to(self.device) for x in xs]
        if cat:
            return torch.cat(xs, axis)
        else:
            return xs  # w/o cat (i.e., the list (not batch) of tensor data).
        
    def pad(self, matrices, pad_value):
        """Ajoute du padding à une liste de matrices
        avec une valeur de padding (ex: 0 ou une grande valeur) pour le traitement par batch.
        Par exemple, pour une liste de matrices [A, B, C],
        cette fonction retourne une nouvelle matrice [A00, 0B0, 00C],
        où 0 est la matrice nulle (i.e., une matrice diagonale par blocs).
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        pad_matrices = torch.full((M, N), pad_value, device=self.device)
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            matrix = torch.FloatTensor(matrix).to(self.device)
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices
    
    
    def basis_matrix(self, atomic_orbitals, distance_matrices, quantum_numbers, fields_coords):
        """Calcule la matrice de base en combinant les fonctions radiales (GTOs) et angulaires (harmoniques sphériques)."""
        n = quantum_numbers
        l = n - 1 # Le nombre quantique azimutal l est dérivé de n
        
        l = l.detach().cpu().tolist()
        
        l = [int(x) for x in l]
        
        # Calcule les harmoniques sphériques Y_lm pour les coordonnées des champs (points dans l'espace)
        Y_lm = spherical_harmonics(l, fields_coords, normalize='True')
        
        # Assure la cohérence des dimensions pour la multiplication matricielle
        Y_lm = Y_lm[:, :fields_coords.shape[0]] 
        
        
        # Récupère les exposants zêta pour chaque orbitale atomique
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        
        # Calcule la partie radiale de la base (type GTO modifié)
        Radials = (distance_matrices**(quantum_numbers-1) * torch.exp(-zetas*distance_matrices**2))
        # Normalise les fonctions radiales
        Radials = F.normalize(Radials, p=2, dim=0)    
        
        # print(torch.sum(torch.t(Radials)[0]**2))  # Vérification de la normalisation
        
        # print("Radials", Radials.shape, "Y_lm", Y_lm.shape) # Vérification des dimensions

        # Combine les parties radiales et angulaires pour obtenir la matrice de base
        GTOs = torch.matmul(Radials, Y_lm)
        
        return GTOs
    
    
    def LCAO(self, inputs):
        """Calcule les orbitales moléculaires par Combinaison Linéaire d'Orbitales Atomiques (LCAO)."""
        (atomic_orbitals, distance_matrices,
         quantum_numbers, atomic_coords, N_electrons, N_fields) = inputs

        """Prépare les données d'entrée pour le traitement par batch (concaténation ou padding)."""
        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.pad(distance_matrices, 1e6)
        quantum_numbers = self.list_to_batch(quantum_numbers, cat=True, axis=1)
        N_electrons = self.list_to_batch(N_electrons)
        atomic_coords = self.list_to_batch(atomic_coords, cat=True, axis=0)

        
        """Normalise les coefficients (embeddings) des orbitales atomiques."""
        coefficients = []
        for AOs in atomic_orbitals:
            coefs = F.normalize(self.coefficient(AOs), 2, 0)
            #print(torch.sum(torch.t(coefs)[0]**2))  # Vérification de la normalisation.
            coefficients.append(coefs)
        coefficients = torch.cat(coefficients)  # Concatène les coefficients pour le batch
        atomic_orbitals = torch.cat(atomic_orbitals)
        

        """Calcul LCAO."""
        
        quantum_numbers = quantum_numbers[0] # Prend les nombres quantiques du premier élément (supposés identiques dans le batch)
        
        # Calcule la matrice de base
        basis_matrix = self.basis_matrix(atomic_orbitals,
                                        distance_matrices, quantum_numbers, atomic_coords)
        #print("basis_matrix", basis_matrix.shape, "coefficients", coefficients.shape)
        # Calcule les orbitales moléculaires par produit matriciel
        molecular_orbitals = torch.matmul(basis_matrix, coefficients)
        

        """Normalise les orbitales moléculaires pour conserver le nombre total d'électrons."""
            
        # Sépare les orbitales moléculaires par molécule dans le batch
        split_MOs = torch.split(molecular_orbitals, N_fields)   # a reverifier si erreur
        normalized_MOs = []
        for N_elec, MOs in zip(N_electrons, split_MOs):
            # Normalise chaque ensemble d'orbitales moléculaires
            MOs = torch.sqrt(N_elec/self.dim) * F.normalize(MOs, 2, 0)
            # print(torch.sum(MOs**2), N_elec)  # Vérification du nombre total d'électrons.
            normalized_MOs.append(MOs)

        # Concatène les orbitales moléculaires normalisées pour le batch
        return torch.cat(normalized_MOs)
    
    
    def functional(self, vectors, layers, operation, axis):
        """Applique le fonctionnel de densité basé sur un réseau de neurones profond (DNN)."""
        for l in range(layers):
            input_vectors = vectors
            # Couche linéaire
            out_vectors = self.W_functional[l](input_vectors)
            # Normalisation par batch
            out_vectors = self.bn_functional[l](out_vectors)        
            # Fonction d'activation PReLU
            vectors = self.prelu(out_vectors)  # Autres fonctions d'activation possible à tester (relu, leaky_relu, selu, silu, mish, gelu, tanh)
            # Couche Dropout pour la régularisation
            out_vectors = self.dropout_functional[l](out_vectors) 
            # Connexion résiduelle si les dimensions correspondent
            if out_vectors.shape == input_vectors.shape:
                vectors = out_vectors + input_vectors

        # Opération finale pour agréger les vecteurs (somme ou moyenne)
        if operation == 'sum':  # pour propriétés des matériaux comme PCE
            vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        if operation == 'mean':  # pour Homo ou Lumo
            vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        # Empile les vecteurs résultants pour former un tenseur
        return torch.stack(vectors)

    def HKmap(self, scalars, layers):
        """Applique la carte de Hohenberg-Kohn basée sur un DNN pour prédire le potentiel à partir de la densité."""
        # Première couche linéaire prenant la densité en entrée
        vectors = self.W_density(scalars)
        # Couches cachées avec activation ReLU
        for l in range(layers):
            vectors = torch.relu(self.W_HK[l](vectors))
        # Couche de sortie pour prédire le potentiel
        return self.W_potential(vectors)
    
    def forward(self, data, train=False, target=None, predict=False):
        """Ecaluation du modèle QDF."""

        # Extrait les données d'entrée
        idx, inputs, N_fields = data[0], data[1:7], data[6]
        
        if predict: # Mode prédiction (pas de calcul de gradient)
            with torch.no_grad():
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,self.layer_functional,self.operation, N_fields)
                PCE_ = self.W_property(final_layer) # Prédiction de la propriété (PCE)
                return idx, PCE_
            
        elif train: # Mode entraînement
            if target == 'PCE':  # Apprentissage supervisé pour l'énergie (PCE)
                PCE = self.list_to_batch(data[7], cat=True, axis=0)  # PCE réel
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,self.layer_functional,self.operation, N_fields)
                PCE_ = self.W_property(final_layer)  # PCE prédit
                #print("LOSS_PCE", F.l1_loss(PCE, PCE_))
                loss = F.l1_loss(PCE, PCE_) # Calcul de la perte L1 (MAE), on pourrait utiliser une perte RMSE aussi mais il faudrait réécrire la fonction de perte de la class Test et trainer
            if target == 'V':  # Apprentissage supervisé pour le potentiel (via HK map)
                V = self.list_to_batch(data[8], cat=True, axis=0)  # Potentiel réel (si disponible)
                molecular_orbitals = self.LCAO(inputs)
                # Calcule la densité électronique à partir des orbitales moléculaires
                densities = torch.sum(molecular_orbitals**2, 1)
                densities = torch.unsqueeze(densities, 1)
                V_ = self.HKmap(densities, self.layer_HK)  # Potentiel prédit par la carte HK
                loss = F.l1_loss(V, V_) # Calcul de la perte L1 (MAE) entre potentiel prédit et réel
            return loss
        
        else:  # Mode test (évaluation, pas de calcul de gradient)
            with torch.no_grad():
                PCE = self.list_to_batch(data[7], cat=True, axis=0) # PCE réel
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,self.layer_functional,self.operation, N_fields)
                PCE_ = self.W_property(final_layer)  # PCE prédit
                PCE_ = PCE_.to(self.device)
                return idx, PCE, PCE_ # Retourne ID, PCE réel, PCE prédit
            
            
            

class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size):
        self.model = model
        # Initialise l'optimiseur Adam avec le taux d'apprentissage donné
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        # Initialise un planificateur pour réduire le taux d'apprentissage toutes les 'step_size' époques
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, lr_decay)  
        
    def optimize(self, loss, optimizer):
        """Effectue une étape d'optimisation (rétropropagation et mise à jour des poids)."""
        optimizer.zero_grad() # Remet à zéro les gradients accumulés
        loss.backward() # Calcule les gradients par rétropropagation
        optimizer.step()  # Met à jour les paramètres du modèle
        
    def train(self, dataloader):
        """Effectue une époque d'entraînement complète sur le dataloader fourni."""
        losses_PCE, losses_V = 0, 0 # Initialise les pertes cumulées
        # Calcule le nombre total d'échantillons dans le dataloader
        n_batches = sum([len(data[0]) for data in dataloader])
        for data in dataloader:
            # Calcule la perte pour la cible 'PCE' (supervisé)
            loss_PCE = self.model.forward(data, train=True, target='PCE')
            # Optimise le modèle en fonction de la perte PCE
            self.optimize(loss_PCE, self.optimizer)
            losses_PCE += loss_PCE.item() # Accumule la perte PCE
            # Calcule la perte pour la cible 'V' (non supervisé via HK map)
            loss_V = self.model.forward(data, train=True, target='V')
            # Optimise le modèle en fonction de la perte V
            self.optimize(loss_V, self.optimizer)
            losses_V += loss_V.item() # Accumule la perte V
            #print("Loss V", loss_V)
        self.scheduler.step()
        #print("n_bacthes", n_batches)
        # Retourne les pertes moyennes pour l'époque
        return losses_PCE/n_batches, losses_V/n_batches




class Tester(object):
    def __init__(self, model):
        self.model = model
        
    def accuracy(self, PCEs, PCEs_, seuil=2.5):
        """Proportion de prédictions avec erreur absolue < seuil."""
        PCEs = np.array(PCEs)
        PCEs_ = np.array(PCEs_)
        correct = np.abs(PCEs - PCEs_) < seuil
        return np.mean(correct)

    def test(self, dataloader, time=False):
        """Évalue le modèle sur le dataloader fourni et calcule la MAE."""
        # Calcule le nombre total d'échantillons
        N = sum([len(data[0]) for data in dataloader])
        IDs, PCEs, PCEs_ = [], [], [] # Listes pour stocker les IDs, valeurs réelles et prédites
        SAE = 0 
        start = timeit.default_timer() # Démarre le chronomètre
        
        for i, data in enumerate(dataloader):
            # Effectue la prédiction en mode test (pas de calcul de gradient)
            idx, PCE, PCE_ = self.model.forward(data)
            # Calcule l'erreur absolue pour le batch
            SAE_batch = torch.sum(torch.abs(PCE - PCE_), 0)
            SAE += SAE_batch # Accumule l'erreur absolue
            IDs += list(idx) # Ajoute les IDs du batch
            PCEs += PCE.tolist() # Ajoute les valeurs réelles du batch
            PCEs_ += PCE_.tolist() # Ajoute les valeurs prédites du batch

            # Estime le temps restant (si time=True)
            if (time is True and i == 0):
                time_elapsed = timeit.default_timer() - start
                minutes = len(dataloader) * time_elapsed / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('La prédiction se terminera dans environ',
                      hours, 'heures', minutes, 'minutes.')

        # Calcule l'erreur absolue moyenne (MAE)
        MAE = (SAE/N).tolist()  
        # Formate la MAE en chaîne de caractères (utile si plusieurs sorties)
        MAE = ','.join([str(m) for m in MAE])  
        
        accuracy = self.accuracy(PCEs, PCEs_, seuil=2.5) # Calcule la précision, on peut changer le seuil pour avoir une meilleure précision (ex: 0.8)
        
        # Crée une chaîne de caractères pour stocker les résultats détaillés
        prediction = 'ID\tCorrect\tPredict\tError\n'
        for idx, PCE, PCE_ in zip(IDs, PCEs, PCEs_):
            # Calcule l'erreur absolue pour chaque échantillon
            error = np.abs(np.array(PCE) - np.array(PCE_))
            error = ','.join([str(e) for e in error]) # Formate l'erreur
            # Formate les valeurs PCE et PCE_
            PCE = str(PCE[0])
            PCE_ = str(PCE_[0])
            # Ajoute la ligne au résultat
            prediction += '\t'.join([idx, PCE, PCE_, error]) + '\n'

        return MAE, prediction, accuracy
    
    def save_result(self, result, filename):
        """Sauvegarde un résultat (ex: MAE) dans un fichier texte."""
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        """Sauvegarde la chaîne de prédiction détaillée dans un fichier texte."""
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        """Sauvegarde les poids du modèle entraîné."""
        torch.save(model.state_dict(), filename)
        
        
        
        
        
class MyDataset(torch.utils.data.Dataset):
    """Classe Dataset pour charger les données depuis des fichiers .npy."""
    def __init__(self, directory):
        self.directory = directory
        # Liste les fichiers .npy dans le répertoire, triés par date de modification
        paths = sorted(Path(self.directory).iterdir(), key=os.path.getmtime)
        # Extrait les noms de fichiers
        self.files = [str(p).strip().split('/')[-1] for p in paths]
        

    def __len__(self):
        """Retourne le nombre total de fichiers (échantillons) dans le dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """Charge et retourne un échantillon de données à partir de son index."""
        # Construit le chemin complet du fichier
        filepath = self.directory + self.files[idx]
        # Gère le cas où le chemin est déjà complet (peut arriver selon la construction de self.files)
        if len(self.files[idx]) > len(self.directory):
             filepath = self.files[idx]
        # Charge le fichier .npy (allow_pickle=True est nécessaire si les arrays contiennent des objets)
        return np.load(filepath, allow_pickle=True)


def mydataloader(dataset, batch_size, num_workers, shuffle=False):
    """Crée un DataLoader PyTorch pour le dataset donné."""
    dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, # Taille du batch
                shuffle=shuffle, # Mélanger les données à chaque époque ?
                num_workers=num_workers, # Nombre de processus pour charger les données en parallèle
                # collate_fn regroupe les échantillons d'un batch
                collate_fn=lambda xs: list(zip(*xs)), 
                pin_memory=True # Accélère le transfert de données vers le GPU si possible
                )
    return dataloader


def objective(params):
    """Fonction objectif à minimiser par hyperopt pour trouver les meilleurs hyperparamètres."""
    # Récupérer les hyperparamètres à tester depuis le dictionnaire 'params'
    dim = int(params['dim'])
    layer_functional = int(params['layer_functional'])
    hidden_HK = int(params['hidden_HK'])
    layer_HK = int(params['layer_HK'])
    lr = params['lr']
    lr_decay = params['lr_decay']
    
    # Créer les datasets d'entraînement et de validation
    dataset_train = MyDataset(params["data"]["train"])
    dataset_val = MyDataset(params["data"]["val"])
    # Limiter la taille du dataset d'entraînement pour accélérer l'optimisation
    dataset_train.files = dataset_train.files[:550]   
    # Créer les dataloaders
    dataloader_train = mydataloader(dataset_train, batch_size=int(params["batch_size"]), num_workers=0, shuffle=True)
    dataloader_val = mydataloader(dataset_val, batch_size=int(params["batch_size"]), num_workers=0)
    
    # Définir le device (GPU si disponible, sinon CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Récupérer d'autres paramètres nécessaires au modèle
    N_orbitals = params["N_orbitals"]
    N_output = params["N_output"]
    operation = params["operation"]
    # Créer le modèle QDF avec les hyperparamètres courants
    model = QuantumDeepField(device, N_orbitals, dim, layer_functional, operation, N_output, hidden_HK, layer_HK).to(device)
    
    # Créer l'entraîneur et le testeur
    trainer = Trainer(model, lr, lr_decay, step_size=params['step_size'])
    tester = Tester(model)
    
    # Entraîner le modèle pendant un nombre fixe d'époques pour évaluer les hyperparamètres
    n_epochs = 20
    for epoch in range(n_epochs):
        loss_E, loss_V = trainer.train(dataloader_train)
        # Évaluer la performance sur l'ensemble de validation à chaque époque (ou à la fin)
        MAE_val, prediction, acc_test = tester.test(dataloader_val)
        
    # Extraire la MAE de validation (peut être une chaîne si plusieurs sorties)
    try:
        mae = float(MAE_val)
    except:
        # Si MAE_val est une chaîne (ex: '0.1,0.2'), calculer la moyenne
        mae = np.mean([float(m) for m in MAE_val.split(',')])
        
    # Afficher les paramètres testés et la MAE obtenue
    print("Params:", params, "=> MAE:", mae)
    # Retourner un dictionnaire attendu par hyperopt, contenant la perte (MAE) à minimiser
    return {'loss': mae, 'status': STATUS_OK}
    

if __name__ == "__main__":
    """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius')
    parser.add_argument('grid_interval')
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    parser.add_argument('hyperparam_optimizer', type=str)
    args = parser.parse_args()
    
    
    dataset = args.dataset
    unit = '(' + dataset.split('_')[-1] + ')'
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    setting = args.setting
    num_workers = args.num_workers
    hyperparam_optimizer = args.hyperparam_optimizer
    
    
    # torch.manual_seed(123) # Cela garantit que les tirages pseudo-aléatoires seront reproductibles pour chaque exécution
    
    # verifier si le GPU est disponible et l'utiliser si c'est le cas
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Il y a " + str(torch.cuda.device_count()) + " GPU(s) disponible.")
        print('Le code utilise un GPU : ' + torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Le code utilise un CPU')
    print('-'*50)
    
    """ Créer les dataloaders pour l'ensemble d'entraînement, de validation et de test."""
    dir_dataset = '../../datasets/' + dataset + '/'
    field = '_'.join([basis_set, 'sphere' , str(radius), grid_interval + 'grid/']) # définition d'un nom pour se reperer
    
    dataset_train = MyDataset(dir_dataset + 'train/train_' + field)
    dataset_val = MyDataset(dir_dataset + 'val/val_' + field)
    dataset_test = MyDataset(dir_dataset + 'test/test_' + field)
    
    with open(dir_dataset + 'orbitaldict_' + dataset + "/" + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)
    
    """ La dimension de sortie pour la régression.
    Lorsque nous apprenons PCE, N_output=1;"""

    N_output = len(dataset_train[0][-2][0])  # on recupere la propriete qu'on veut prédire
    
    if hyperparam_optimizer.lower() == "true":
    
        print("Commence l'optimization des hyperparamètres.\n'")
        
        # Définir l'espace d'hyperparamètres
        space = {
            'dim': hp.quniform('dim', 100, 500, 50),
            'layer_functional': hp.quniform('layer_functional', 1, 5, 1),
            'hidden_HK': hp.quniform('hidden_HK', 50, 300, 50),
            'layer_HK': hp.quniform('layer_HK', 1, 4, 1),
            'lr': hp.loguniform('lr', -10, -3),
            'lr_decay': hp.uniform('lr_decay', 0.5, 0.99),
            'step_size': hp.quniform('step_size', 2, 24, 2),
            'iteration': hp.quniform('iteration', 200, 400, 50),
            'setting': setting,
            'operation': operation,
            'N_orbitals': len(orbital_dict),
            'N_output': N_output,
            'batch_size': hp.quniform('batch_size', 2, 5, 1),
            'data' : {
                'train': dir_dataset + 'train/train_' + field,
                'val': dir_dataset + 'val/val_' + field,
                'test': dir_dataset + 'test/test_' + field
            }
        }
        
        trials = Trials()
        
        #best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=80, trials=trials)
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials) # agrandir pour plus de résultats
        
        print("\n Best hyperparameters:", best)
        print("Best MAE:", trials.best_trial['result']['loss'])
    
        print('-'*50)
    
    """Crée le model."""
    if  hyperparam_optimizer.lower() == "true":
        dataloader_train = mydataloader(dataset_train, int(best["batch_size"]), num_workers,shuffle=True)  # charge les données d’entraînement
        dataloader_val = mydataloader(dataset_val, int(best["batch_size"]), num_workers)
        dataloader_test = mydataloader(dataset_test, int(best["batch_size"]), num_workers)
        
        print('training samples: ', len(dataset_train))
        print('validation samples: ', len(dataset_val))
        print('test samples: ', len(dataset_test))
        print('-'*50)

        model = QuantumDeepField(device, N_orbitals,
                int(best["dim"]), int(best["layer_functional"]), operation, N_output,
                int(best["hidden_HK"]), int(best["layer_HK"])).to(device)


        print("Ici nous avons le model \n" ,model)
        
    
        trainer = Trainer(model, best["lr"], best["lr_decay"], int(best["step_size"]))
        tester = Tester(model)
        
        iteration = int(best["iteration"])
        
        with open("best_hyperparameters.txt", "w") as f:
            f.write("Best hyperparameters:\n")
            for param_name, param_value in best.items():
                if param_name == "lr" or param_name == "lr_decay":   
                    f.write(f"{param_name}={param_value:.6f}\n")
                else :
                    f.write(f"{param_name}={int(param_value)}\n")
            f.write(f"Best MAE: {trials.best_trial['result']['loss']}\n")

        print('Nombres de paramètres totaux:',
                sum([np.prod(p.size()) for p in model.parameters()]))
        
        
        setting = f"{dataset}--{basis_set}--radius{radius}--grid_interval{grid_interval}--dim{int(best['dim'])}--layer_functional{int(best['layer_functional'])}--hidden_HK{int(best['hidden_HK'])}--layer_HK{int(best['layer_HK'])}--{operation}--batch_size{int(best['batch_size'])}--lr{best['lr']:.6f}--lr_decay{best['lr_decay']:.6f}--step_size{int(best['step_size'])}--iteration{int(best['iteration'])}"
        
    else: 
        dataloader_train = mydataloader(dataset_train, batch_size, num_workers,shuffle=True)  # charge les données d’entraînement
        dataloader_val = mydataloader(dataset_val, batch_size, num_workers)
        dataloader_test = mydataloader(dataset_test, batch_size, num_workers)
        
        print('training samples: ', len(dataset_train))
        print('validation samples: ', len(dataset_val))
        print('test samples: ', len(dataset_test))
        print('-'*50)

        model = QuantumDeepField(device, N_orbitals,
                dim, layer_functional, operation, N_output,
                hidden_HK, layer_HK).to(device)


        print(" Affichage du model \n" ,model)
        

        trainer = Trainer(model, lr, lr_decay, step_size)
        tester = Tester(model)

        print('Nombres de paramètres totaux:',
                sum([np.prod(p.size()) for p in model.parameters()]))
    
    print('-'*50)
    
    
    """Output files."""
    file_result = '../../datasets/output/' + setting + '.txt'
    os.makedirs(os.path.dirname(file_result), exist_ok=True)
    result = ('Epoch\tTime(sec)\tLoss_PCE\tLoss_V\tMAE_val' + unit + '\tMAE_test' + unit + '\tAcc_val' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../../datasets/output/pred--' + setting + '.txt'
    file_model = '../../datasets/output/model--' + dataset +'.pth'
    os.makedirs(os.path.dirname(file_prediction), exist_ok=True)
    os.makedirs(os.path.dirname(file_model), exist_ok=True)
    
    print('-'*50)
    
    print("Commence l'entrainement du modèle QDF avec le dataset", dataset, "\n"
            "Le résultat de l'entrainement est affiché dans ce terminal à chaque époque.\n"
            "Le résultat, la prédiction et le modèle entraîné "
            "sont enregistrés dans le répertoire de sortie.\n"
            "Attendez un moment...")

    start = timeit.default_timer()
    
    best_val_mae = float('inf')  # On démarre avec une valeur arbitrairement haute

    for epoch in range(iteration):
        loss_E, loss_V = trainer.train(dataloader_train)
        MAE_val, pred, acc_val = tester.test(dataloader_val)
        MAE_test, prediction, acc_test = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        if epoch == 0:
            minutes = iteration * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('Le training finira dans environ',
                    hours, 'heures', minutes, 'minutes.')
            print('-'*50)

        result = '\t'.join(map(str, [epoch, time, loss_E, loss_V, MAE_val, MAE_test, acc_val]))
        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        # Vérification et sauvegarde du “meilleur” modèle
        if float(MAE_val) < best_val_mae:
            best_val_mae = float(MAE_val)
            
            tester.save_model(model, file_model)  # On écrase l’ancien modèle
            print("A l'epoch", epoch, "le model est sauvegardé.")
            print("La meilleure MAE est pour la validation est ", best_val_mae)

    print('Entrainement terminé.')
    