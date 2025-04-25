import argparse
import pickle
import sys

import torch

sys.path.append('../')
from train_model import QDF_SM
import numpy as np



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_trained')
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
    parser.add_argument('dataset_predict')
    args = parser.parse_args()
    dataset_trained = args.dataset_trained
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
    dataset_predict = args.dataset_predict
    
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("There is " + str(torch.cuda.device_count()) + " GPU(s) available.")
        print('The code uses a GPU : ' + torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-'*50)
    
    
    dir_trained = '../../datasets/' + dataset_trained + '/'
    dir_predict = '../../datasets/' + dataset_predict + '/'
    
    field = '_'.join([basis_set, 'sphere', str(radius), str(grid_interval) + 'grid/'])
    
    if dataset_predict == 'data_test':  # on charge le daaset pour la demo
        dataset_test = QDF_SM.MyDataset(dir_predict + '/demo' + '/demo_' + basis_set)  
    else:
        dataset_test = QDF_SM.MyDataset(dir_predict + '/test/test_' + field)
    dataloader_test = QDF_SM.mydataloader(dataset_test, batch_size=batch_size,num_workers=num_workers)
    
    with open(dir_trained + 'orbitaldict_' + dataset_trained + "/" + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    
    
    N_output = 1
    
    model = QDF_SM.QuantumDeepField(device, N_orbitals,
                                dim, layer_functional, operation, N_output,
                                hidden_HK, layer_HK).to(device)
    
    print("Here we have the model \n" ,model)
        


    print('# of model parameters:',
            sum([np.prod(p.size()) for p in model.parameters()]))

    print('-'*50)


    model.load_state_dict(torch.load('../../datasets/output/model--' + dataset_trained + ".pth", map_location=device))
    
    print('Commencez la prédiction pour le jeu de données', dataset_predict, '\n'
          'en utilisant le modèle pré-entraîné avec le jeu de données', dataset_trained, '\n'
          'Le résultat de la prédiction est enregistré dans le répertoire de sortie.\n'
          'Attendez un moment...')
    
    model.eval()
    
    prediction = 'ID\tPredict\n'

    for data in dataloader_test:
        
        idx, PCE_ = model.forward(data, predict=True)
        for i in range(len(idx)):
            prediction += str(idx[i]) + '\t' + str(PCE_[i][0].item()) + '\n'

    print(prediction)

    print('Entrainement terminé.')
