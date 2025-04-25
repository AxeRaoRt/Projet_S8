
import argparse
import pickle
import sys
import os
import timeit
import numpy as np
import torch

#sys.path.append('../')
import QDF_SM 
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK



if __name__ == "__main__":

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
    parser.add_argument('predataset')
    
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
    predataset=args.predataset
    
    
    # verifier si le GPU est disponible et l'utiliser si c'est le cas
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Il y a " + str(torch.cuda.device_count()) + " GPU(s) disponible.")
        print('Le code utilise un GPU : ' + torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('Le code utilise un CPU')
    print('-'*50)
    
    """ Créer les dataloaders de l'ensemble d'entraînement, de validation et de test."""
    dir_dataset = '../../datasets/' + dataset + '/'
    field = '_'.join([basis_set, 'sphere' , str(radius), grid_interval + 'grid/'])
    
    dataset_train = QDF_SM.MyDataset(dir_dataset + 'train/train_' + field)
    dataset_val = QDF_SM.MyDataset(dir_dataset + 'val/val_' + field)
    dataset_test = QDF_SM.MyDataset(dir_dataset + 'test/test_' + field)

    # on récupère le dictionnaire d'orbital atomiques
    with open(dir_dataset + 'orbitaldict_' + dataset + "/" + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)
    
    N_output = len(dataset_test[0][-2][0])
    
    """Fichiers de sortie"""
    file_result = '../../datasets/output/' + setting + '.txt'
    os.makedirs(os.path.dirname(file_result), exist_ok=True)
    result = ('Epoch\tTime(sec)\tLoss_PCE\tLoss_V\tMAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../../datasets/output/pred--' + setting + '.txt'
    file_model = '../../datasets/output/model--' + dataset +'.pth'
    os.makedirs(os.path.dirname(file_prediction), exist_ok=True)
    os.makedirs(os.path.dirname(file_model), exist_ok=True)
    
    print('-'*50)
    
    dataloader_train = QDF_SM.mydataloader(dataset_train, batch_size, num_workers,shuffle=True)  # charge les données d’entraînement
    dataloader_val = QDF_SM.mydataloader(dataset_val, batch_size, num_workers)
    dataloader_test = QDF_SM.mydataloader(dataset_test, batch_size, num_workers)
    
    print('training samples: ', len(dataset_train))
    print('validation samples: ', len(dataset_val))
    print('test samples: ', len(dataset_test))
    print('-'*50)

    model = QDF_SM.QuantumDeepField(device, N_orbitals,
            dim, layer_functional, operation, N_output,
            hidden_HK, layer_HK).to(device)   # on recréee le modèle pour le fine-tuning

    state_dict = torch.load('../../datasets/output/model--' + predataset + ".pth", map_location=device)

    """ On supprime les paramètres incompatibles 
    car il y a un changement du dictionnaire d'orbitales atomiques 
    donc certains paramètres sont différents """
    
    to_remove = []  
    for k, v in state_dict.items():
        if k in model.state_dict():
            if model.state_dict()[k].shape != v.shape:
                print(f"Skip loading parameter: {k} (checkpoint shape: {v.shape}, model shape: {model.state_dict()[k].shape})")
                to_remove.append(k)
    for k in to_remove:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)  # charge le modèle pré-entraîné
    
    print("Ici on a le modèle \n", model)

    trainer = QDF_SM.Trainer(model, lr, lr_decay, step_size)
    tester = QDF_SM.Tester(model)

    print('Nombres de paramètres totaux:',
                sum([np.prod(p.size()) for p in model.parameters()]))
    
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
    
    