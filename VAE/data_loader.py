import numpy as np  # Pour les tableaux numériques et le padding
import selfies as sf  # Pour manipuler les chaînes SELFIES

# === Fonction pour encoder un seul SMILES en one-hot ===
def smile_to_hot(smile, largest_smile_len, alphabet):
    """
    Convertit une chaîne SMILES unique en encodage one-hot.
    """
    # Associe chaque caractère de l'alphabet à un index entier
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # Complète le SMILES avec des espaces pour atteindre la longueur maximale
    smile += ' ' * (largest_smile_len - len(smile))

    # Encode chaque caractère du SMILES en entier
    integer_encoded = [char_to_int[char] for char in smile]

    # Encode chaque entier en vecteur one-hot
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]  # vecteur rempli de 0
        letter[value] = 1  # place un 1 à la position correspondant au caractère
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)


# === Fonction pour encoder plusieurs SMILES à la fois ===
def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """
    Convertit une liste de SMILES en encodage one-hot.
    """
    hot_list = []
    for s in smiles_list:
        _, onehot_encoded = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)

    # Récupère la plus grande longueur parmi les encodages pour homogénéiser
    max_len = max(len(hot) for hot in hot_list)

    # Applique du padding à chaque vecteur pour qu'ils aient tous la même longueur
    padded_hot_list = [np.pad(hot, ((0, max_len - len(hot)), (0, 0)), mode='constant') for hot in hot_list]

    return np.array(padded_hot_list)


# === Fonction pour encoder un seul SELFIES en one-hot ===
def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """
    Convertit une chaîne SELFIES unique en encodage one-hot.
    """
    # Associe chaque symbole SELFIES à un index entier
    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # Ajoute des [nop] pour compléter la chaîne SELFIES jusqu'à la longueur max
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # Découpe la chaîne SELFIES en symboles
    symbol_list = sf.split_selfies(selfie)

    # Encode chaque symbole en entier
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # Encode chaque entier en vecteur one-hot
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)  # vecteur de 0
        letter[index] = 1  # 1 à la position du symbole
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)


# === Fonction pour encoder plusieurs SELFIES en one-hot ===
def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """
    Convertit une liste de SELFIES en encodage one-hot.
    """
    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)

    # Trouve la longueur max des encodages one-hot
    max_len = max(len(hot) for hot in hot_list)

    # Applique du padding sur chaque encodage pour aligner les dimensions
    padded_hot_list = [np.pad(hot, ((0, max_len - len(hot)), (0, 0)), mode='constant') for hot in hot_list]

    return np.array(padded_hot_list)
