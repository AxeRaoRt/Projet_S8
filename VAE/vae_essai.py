import os  # for file and directory operations
import sys  # for system-specific parameters and functions
import time  # for measuring time and delays

import numpy as np  # for numerical operations
import pandas as pd  # for data manipulation and analysis
import torch  # for tensor operations and deep learning
import torch.nn.functional as F  # for neural network functions like activation, loss
import yaml  # for reading/writing .yaml configuration files
from pathlib import Path  # for object-oriented file paths
from rdkit import rdBase  # base RDKit module
from rdkit.Chem import MolFromSmiles, Draw  # for molecule parsing and drawing
from torch import nn  # for building neural network layers
import selfies as sf  # for handling SELFIES molecular representation
from rdkit import Chem  # optional, for SMILES validation and molecule operations

import selfies as sf  # (duplicate) for handling SELFIES
from data_loader import multiple_selfies_to_hot, multiple_smile_to_hot  # for converting molecules to one-hot format

 
rdBase.DisableLog("rdApp.error")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def _make_dir(directory):
    # Crée un dossier s'il n'existe pas
    os.makedirs(directory, exist_ok=True)

def save_models(encoder, decoder, epoch):
    # Sauvegarde les poids de l'encodeur et du décodeur
    out_dir = f"./saved_models/{epoch}"
    _make_dir(out_dir)
    torch.save(encoder.state_dict(), f"{out_dir}/E.pth")
    torch.save(decoder.state_dict(), f"{out_dir}/D.pth")

def load_models(epoch):
    # Charge les poids de l'encodeur et du décodeur
    out_dir = f"./saved_models/{epoch}"
    encoder = VAEEncoderGumbel(in_dimension=len_max_mol_one_hot, categorical_dimension=len_alphabet, **encoder_parameter).to(device)
    decoder = VAEDecoderGumbel(**encoder_parameter, categorical_dimension=len_alphabet, out_dimension=len_max_mol_one_hot).to(device)

    torch.serialization.add_safe_globals([VAEEncoderGumbel, VAEDecoderGumbel])
    encoder_path = f"{out_dir}/E.pth"
    decoder_path = f"{out_dir}/D.pth"
    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        # Arrête si les fichiers de modèle sont manquants
        print(f"Warning: Model files not found: {encoder_path} or {decoder_path}. Please ensure the model files are in the correct directory.")
        sys.exit(1)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))
    return encoder, decoder

class VAEEncoder(nn.Module):
    # Encodeur VAE classique
    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d, latent_dimension):
        super(VAEEncoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU()
        )
        self.encode_mu = nn.Linear(layer_3d, latent_dimension)
        self.encode_log_var = nn.Linear(layer_3d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        # Échantillonnage via le trick de reparamétrisation
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # Applique l'encodeur et retourne z, mu, log_var
        h1 = self.encode_nn(x)
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class VAEDecoder(nn.Module):
    # Décodeur VAE avec RNN
    def __init__(self, latent_dimension, gru_stack_size, gru_neurons_num, out_dimension):
        super(VAEDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False)
        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        # Initialise l'état caché du RNN
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size, self.gru_neurons_num)

    def forward(self, z, hidden):
        # Décode la séquence à partir de z
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)
        return decoded, hidden

def sample_gumbel(shape, eps=1e-20):
    # Génère du bruit gumbel
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    # Applique un échantillonnage gumbel-softmax
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

class VAEEncoderGumbel(nn.Module):
    # Encodeur VAE avec variables gumbel
    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d, categorical_dimension, latent_dimension):
        super(VAEEncoderGumbel, self).__init__()
        self.latent_dimension = latent_dimension
        self.categorical_dimension = categorical_dimension
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_3d),
            nn.ReLU(),
            nn.Linear(layer_3d, latent_dimension * categorical_dimension),
            nn.ReLU()
        )

    def forward(self, x):
        # Encode en logits pour gumbel softmax
        q = self.encode_nn(x)
        return q

class VAEDecoderGumbel(nn.Module):
    # Décodeur VAE utilisant Gumbel-Softmax
    def __init__(self, latent_dimension, layer_1d, layer_2d, layer_3d, categorical_dimension, out_dimension):
        super(VAEDecoderGumbel, self).__init__()
        self.latent_dimension = latent_dimension
        self.categorical_dimension = categorical_dimension
        self.decode_nn = nn.Sequential(
            nn.Linear(latent_dimension * categorical_dimension, layer_3d),
            nn.ReLU(),
            nn.Linear(layer_3d, layer_2d),
            nn.ReLU(),
            nn.Linear(layer_2d, layer_1d),
            nn.ReLU(),
            nn.Linear(layer_1d, out_dimension),
            nn.Sigmoid()
        )

    def gumbel_softmax(self, logits, temperature, hard=False):
        # Applique gumbel softmax différentiable
        y = gumbel_softmax_sample(logits, temperature)
        if not hard:
            return y.view(-1, self.latent_dimension * self.categorical_dimension)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, self.latent_dimension * self.categorical_dimension)

    def forward(self, q, temp, hard):
        # Décode depuis logits gumbel
        q_y = q.view(q.size(0), self.latent_dimension, self.categorical_dimension)
        z = self.gumbel_softmax(q_y, temp, hard)
        return self.decode_nn(z), F.softmax(q_y, dim=-1).reshape(*q.size())

def is_correct_smiles(smiles):
    # Vérifie si un SMILES est valide
    if smiles == "":
        return False
    try:
        return MolFromSmiles(smiles, sanitize=True) is not None
    except Exception:
        return False

def sample_latent_space(vae_encoder, vae_decoder, sample_len):
    # Génère des échantillons depuis l’espace latent
    vae_encoder.eval()
    vae_decoder.eval()
    gathered_atoms = []
    fancy_latent_point = torch.randn(1, 1, vae_encoder.latent_dimension, device=device)
    hidden = vae_decoder.init_hidden()
    for _ in range(sample_len):
        out_one_hot, hidden = vae_decoder(fancy_latent_point, hidden)
        out_one_hot = out_one_hot.flatten().detach()
        soft = nn.Softmax(0)
        out_one_hot = soft(out_one_hot)
        out_index = out_one_hot.argmax(0)
        gathered_atoms.append(out_index.data.cpu().tolist())
    vae_encoder.train()
    vae_decoder.train()
    return gathered_atoms

def latent_space_quality(vae_encoder, vae_decoder, type_of_encoding, alphabet, sample_num, sample_len):
    # Évalue la qualité de l’espace latent (validité des molécules)
    total_correct = 0
    all_correct_molecules = set()
    print(f"latent_space_quality: Take {sample_num} samples from the latent space")
    for _ in range(1, sample_num + 1):
        molecule_pre = ""
        for i in sample_latent_space(vae_encoder, vae_decoder, sample_len):
            molecule_pre += alphabet[i]
        molecule = molecule_pre.replace(" ", "")
        if type_of_encoding == 1:
            molecule = sf.decoder(molecule)
        if is_correct_smiles(molecule):
            total_correct += 1
            all_correct_molecules.add(molecule)
    return total_correct, len(all_correct_molecules)

 
# Calcule la qualité de reconstruction sur un échantillon du jeu de validation
def quality_in_valid_set(vae_encoder, vae_decoder, data_valid, batch_size):
    data_valid = data_valid[torch.randperm(data_valid.size()[0])]
    num_batches_valid = len(data_valid) // batch_size
    quality_list = []
    for batch_iteration in range(min(25, num_batches_valid)):
        start_idx = batch_iteration * batch_size
        stop_idx = (batch_iteration + 1) * batch_size
        batch = data_valid[start_idx: stop_idx]
        _, trg_len, _ = batch.size()
        inp_flat_one_hot = batch.flatten(start_dim=1)
        latent_points, mus, log_vars = vae_encoder(inp_flat_one_hot)  # encodeur
        latent_points = latent_points.unsqueeze(0)
        hidden = vae_decoder.init_hidden(batch_size=batch_size)  # initialise l'état caché
        out_one_hot = torch.zeros_like(batch, device=device)
        for seq_index in range(trg_len):
            out_one_hot_line, hidden = vae_decoder(latent_points, hidden)  # décodeur
            out_one_hot[:, seq_index, :] = out_one_hot_line[0]
        quality = compute_recon_quality(batch, out_one_hot)  # mesure qualité reconstruction
        quality_list.append(quality)
    return np.mean(quality_list).item()

# Convertit une chaîne SELFIES en image de molécule
def selfies2image(s):
    mol = MolFromSmiles(sf.decoder(s), sanitize=True)
    return Draw.MolToImage(mol)

# Entraîne le VAE avec Gumbel-Softmax
def train_model_gumbel(vae_encoder, vae_decoder, data_train, data_valid, num_epochs, batch_size, lr_enc, lr_dec, sample_num, sample_len, alphabet, type_of_encoding, categorical_dimension, temp, hard, temp_min, anneal_rate, logger=None):
    print("num_epochs: ", num_epochs)
    int_to_symbol = dict((i, c) for i, c in enumerate(alphabet))  # dictionnaire index → symbole
    optimizer_encoder = torch.optim.Adam(vae_encoder.parameters(), lr=lr_enc)
    optimizer_decoder = torch.optim.Adam(vae_decoder.parameters(), lr=lr_dec)
    data_train = data_train.clone().detach().to(device)
    num_batches_train = len(data_train) // batch_size
    quality_valid_list = [0, 0, 0, 0]
    for epoch in range(num_epochs):
        data_train = data_train[torch.randperm(data_train.size()[0])]  # mélange les données
        start = time.time()
        for batch_iteration in range(num_batches_train):
            start_idx = batch_iteration * batch_size
            stop_idx = (batch_iteration + 1) * batch_size
            batch = data_train[start_idx: stop_idx]
            inp_flat_one_hot = batch.flatten(start_dim=1)
            q = vae_encoder(inp_flat_one_hot)  # encode
            out_one_hot, qy = vae_decoder(q, temp, hard)  # decode avec Gumbel
            loss = compute_elbo_loss_gumbel(inp_flat_one_hot, out_one_hot, qy, categorical_dimension, logger=logger)
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_encoder.step()
            optimizer_decoder.step()
            if batch_iteration % 100 == 1:
                temp = np.maximum(temp * np.exp(-anneal_rate * batch_iteration), temp_min)  # diminue la température
            if batch_iteration % 30 == 0:
                end = time.time()
                quality_train = compute_recon_quality(batch, out_one_hot.view(batch.size()))
                target = batch[0]
                generated = out_one_hot[0].view(target.size())
                target_indices = target.reshape(-1, target.shape[1]).argmax(1)
                generated_indices = generated.reshape(-1, generated.shape[1]).argmax(1)
                target_selfies = sf.encoding_to_selfies(np.array(target_indices.cpu()), int_to_symbol, "label")
                generated_selfies = sf.encoding_to_selfies(np.array(generated_indices.cpu()), int_to_symbol, "label")
                print(f"\nTarget:     {target_selfies}")
                print(f"Generated:  {generated_selfies}\n")
                if logger:
                    logger.log({
                        "loss": loss.item(),
                        "quality_train": quality_train,
                        "quality_valid": quality_valid,
                        "predicted": [
                            wandb.Image(selfies2image(target_selfies), caption=target_selfies),
                            wandb.Image(selfies2image(generated_selfies), caption=generated_selfies)
                        ]
                    })
                start = time.time()
        quality_valid = 0.0
        quality_valid_list.append(quality_valid)
        quality_increase = len(quality_valid_list) - np.argmax(quality_valid_list)
        if quality_increase == 1 and quality_valid_list[-1] > 50.:
            corr, unique = latent_space_quality(vae_encoder, vae_decoder, type_of_encoding, alphabet, sample_num, sample_len)  # diversité et corrélation
        else:
            corr, unique = -1., -1.
        
        if quality_valid_list[-1] < 70. and epoch > 200:  # arrêt anticipé
            break
        if epoch > 0 and epoch % 20 == 0:
            save_models(vae_encoder, vae_decoder, epoch)  # sauvegarde des modèles

# Calcule la perte ELBO pour un VAE Gumbel
def compute_elbo_loss_gumbel(x, x_hat, qy, categorical_dim, logger=None):
    criterion = torch.nn.BCELoss(size_average=False)  # erreur binaire
    recon_loss = criterion(x_hat, x)
    log_ratio = torch.log(qy * categorical_dim + 1e-20)
    kld = torch.sum(qy * log_ratio, dim=-1).mean()  # divergence KL
    if logger:
        logger.log({
            "reconstruction_loss": recon_loss.item(),
            "kld": kld.item(),
        })
    return recon_loss + kld

# Noyau gaussien utilisé pour le MMD
def gaussian_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.view(x_size, 1, dim)
    y = y.view(1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)

# Calcule la divergence MMD entre deux distributions
def compute_mmd(x, y):
    x = x.squeeze()
    y = y.squeeze()
    x_kernel = gaussian_kernel(x, x)
    y_kernel = gaussian_kernel(y, y)
    xy_kernel = gaussian_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

# Calcule la qualité de reconstruction en comparant les symboles générés
def compute_recon_quality(x, x_hat):
    x_indices = x.reshape(-1, x.shape[2]).argmax(1)
    x_hat_indices = x_hat.reshape(-1, x_hat.shape[2]).argmax(1)
    differences = 1. - torch.abs(x_hat_indices - x_indices)
    differences = torch.clamp(differences, min=0., max=1.).double()
    quality = 100. * torch.mean(differences)
    return quality.detach().cpu().numpy()

# Charge et convertit un fichier CSV contenant des SMILES en SELFIES + métadonnées
def get_selfie_and_smiles_encodings_for_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_csv(file_path)
    smiles_list = np.asanyarray(df.SMILES)
    smiles_alphabet = list(set("".join(smiles_list)))
    smiles_alphabet.append(" ")
    largest_smiles_len = len(max(smiles_list, key=len))
    print("--> Translating SMILES to SELFIES...")
    selfies_list = list(map(sf.encoder, smiles_list))
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add("[nop]")
    all_selfies_symbols.add(".")
    selfies_alphabet = list(all_selfies_symbols)
    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
    print("Finished translating SMILES to SELFIES.")
    return selfies_list, selfies_alphabet, largest_selfies_len, smiles_list, smiles_alphabet, largest_smiles_len

# Partie principale : chargement des paramètres et des données
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    settings_file_path = project_dir.joinpath("settings.yml")
    if os.path.exists(settings_file_path):
        settings = yaml.safe_load(open(settings_file_path, "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
        sys.exit(1)

    print(f"Using device: {device}")
    print("--> Acquiring data...")
    type_of_encoding = settings["data"]["type_of_encoding"]
    file_name_smiles = settings["data"]["smiles_file"]
    print("Finished acquiring data.")
 

# Préparation des données selon le type d'encodage (SMILES ou SELFIES)
if type_of_encoding == 0:
    print("Representation: SMILES")
    _, _, _, encoding_list, encoding_alphabet, largest_molecule_len = get_selfie_and_smiles_encodings_for_dataset("../"+file_name_smiles)
    print("--> Creating one-hot encoding...")
    data = multiple_smile_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    print("Finished creating one-hot encoding.")
elif type_of_encoding == 1:
    print("Representation: SELFIES")
    encoding_list, encoding_alphabet, largest_molecule_len, _, _, _ = get_selfie_and_smiles_encodings_for_dataset("../"+file_name_smiles)
    largest_molecule_len = 250
    print(encoding_alphabet)
    print(largest_molecule_len)
    print("--> Creating one-hot encoding...")
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    print("Finished creating one-hot encoding.")
else:
    print("type_of_encoding not in {0, 1}.")
    sys.exit(1)

# Extraction des dimensions pour initialiser les modèles
len_max_molec = data.shape[1]
len_alphabet = data.shape[2]
len_max_mol_one_hot = len_max_molec * len_alphabet
print(f"Alphabet has {len_alphabet} letters, largest molecule is {len_max_molec} letters.")

# Lecture des paramètres depuis settings.yml
data_parameters = settings["data"]
batch_size = data_parameters["batch_size"]
encoder_parameter = settings["encoder"]
decoder_parameter = settings["decoder"]
training_parameters = settings["training"]

# Instanciation du VAE encodeur/décodeur avec Gumbel-Softmax
vae_encoder = VAEEncoderGumbel(in_dimension=len_max_mol_one_hot, categorical_dimension=len_alphabet, **encoder_parameter).to(device)
vae_decoder = VAEDecoderGumbel(**encoder_parameter, categorical_dimension=len_alphabet, out_dimension=len_max_mol_one_hot).to(device)

# Chargement des modèles pré-entraînés si disponibles
if training_parameters.get("pretrained_model"):
    encoder, decoder = load_models(training_parameters["pretrained_model"])
    encoder_dict = {k: v for k, v in encoder.state_dict().items() if k in vae_encoder.state_dict() and encoder.state_dict()[k].size() == vae_encoder.state_dict()[k].size()}
    decoder_dict = {k: v for k, v in decoder.state_dict().items() if k in vae_decoder.state_dict() and decoder.state_dict()[k].size() == vae_decoder.state_dict()[k].size()}
    vae_encoder.load_state_dict(encoder_dict, strict=False)
    vae_decoder.load_state_dict(decoder_dict, strict=False)

# Découpage des données en train/validation et lancement de l'entraînement
print("*" * 15, ": -->", device)
data = torch.tensor(data, dtype=torch.float).to(device)
train_valid_test_size = [0.5, 0.5, 0.0]
data = data[torch.randperm(data.size()[0])]
idx_train_val = int(len(data) * train_valid_test_size[0])
idx_val_test = idx_train_val + int(len(data) * train_valid_test_size[1])
data_train = data[0:idx_train_val]
data_valid = data[idx_train_val:idx_val_test]
print("start training")
training_parameters.pop("batch_size", None)
train_model_gumbel(**training_parameters, vae_encoder=vae_encoder, vae_decoder=vae_decoder, batch_size=batch_size, data_train=data_train, data_valid=data_valid, alphabet=encoding_alphabet, type_of_encoding=type_of_encoding, sample_len=len_max_molec, categorical_dimension=len_alphabet, logger=False)
torch.cuda.empty_cache()

# Génération de nouvelles molécules à partir de fichiers d'encodage
smiles = []
encoder, decoder = load_models(119)
encoder.to(device)
decoder.to(device)

# Boucle sur les fichiers d'entrée (encodés)
for x in range(195):
    file_path = f'datasets/datai/{x}.txt'
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist. Please ensure the dataset files are in the correct directory.")
        sys.exit(1)
    
    # Chargement et encodage des données SELFIES
    encoding_list, _, _, _, _, _ = get_selfie_and_smiles_encodings_for_dataset(file_path)
    data = multiple_selfies_to_hot(encoding_list, largest_molecule_len, encoding_alphabet)
    data = torch.FloatTensor(data)
    inp_flat_one_hot = data.flatten(start_dim=1).to(device)
    
    # Encodage latent puis décodage avec le VAE
    q = encoder(inp_flat_one_hot).to(device)
    out_one_hot, qy = decoder(q, temp=1.0, hard=False)
    
    # Reconstruction en indices puis en SELFIES puis SMILES
    for a in range(len(data)):
        target = data[a]
        generated = out_one_hot[a].view(target.size())
        generated_indices = generated.reshape(-1, generated.shape[1]).argmax(1)
        int_to_symbol = dict((i, c) for i, c in enumerate(encoding_alphabet))
        smi = clean_and_decode_selfies(generated_indices, encoding_alphabet)
        smiles.append(smi)

    # Sauvegarde des SMILES générés dans un fichier CSV
    smiles_df = pd.DataFrame(smiles)
    smiles_df.to_csv(f'datasets/datao/{x}.csv', index=False)
    smiles = []
    torch.cuda.empty_cache()

# Fonction pour nettoyer et convertir un encodage SELFIES en SMILES
def clean_and_decode_selfies(generated_indices, encoding_alphabet):
    # Mapping int → symbole SELFIES
    int_to_symbol = {i: c for i, c in enumerate(encoding_alphabet)}

    # Décodage brut en SELFIES
    selfies = sf.encoding_to_selfies(np.array(generated_indices.cpu()), int_to_symbol, "label")
    
    # Nettoyage : suppression des [nop], des points éventuels
    selfies_clean = selfies.replace("[nop]", "").replace(".", "")
    
    # Tentative de décodage SELFIES → SMILES
    try:
        smiles = sf.decoder(selfies_clean)
        # Validation avec RDKit (optionnel mais utile)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print("Décodage SELFIES réussi mais SMILES invalide.")
                return "", selfies_clean
    except sf.DecoderError:
        print("Erreur de décodage SELFIES.")
        smiles = ""
    
    return smiles, selfies_clean


       