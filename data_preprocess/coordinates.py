from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import numpy as np
import pandas as pd
import sys
import os
import argparse

def generate_coordinates(filepath, output_file):
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas.")

    df = pd.read_csv(file_path)
    smiles_list = df["SMILES"].tolist() # Limiter à 2 molécules pour la démo mais on peut l'élargir si le gpu est plus puissant
    
    try:
        property = df["PCE"].tolist()
    except KeyError:
        property = ""
    
    file = open(output_file, "w", encoding="utf-8")
    
    for ind, smiles in enumerate(smiles_list):
        print(f"Processing molecule {ind + 1}...")
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                print(f"Erreur : Impossible de convertir SMILES index {ind}")
                continue
        
            m3d = mol
            params = AllChem.ETKDGv3()
            E = AllChem.EmbedMolecule(m3d, params=params)
        
            if E == -1:
                    E = AllChem.EmbedMolecule(m3d, useRandomCoords=True, ignoreSmoothingFailures=True)
                    AllChem.MMFFOptimizeMolecule(m3d, maxIters=10000)
            
            
            conformer = m3d.GetConformer()
            file.write('pce_'+ str(ind)+'\n')
            # Écriture des résultats dans le fichier
            for i in range(m3d.GetNumAtoms()):
                pos = conformer.GetAtomPosition(i)
                atom_symbol = m3d.GetAtomWithIdx(i).GetSymbol()
                file.write(f"{atom_symbol}    {pos.x:.6f}    {pos.y:.6f}     {pos.z:.6f}\n")
            if property == "":
                file.write(f"")
            else :
                file.write(f"{property[ind]}\n")
            file.write("\n\n")
            
        except Exception as e:
                print(f"Erreur lors du traitement de {smiles} : {e}")
                continue
    file.close()
    print(f"Fichier généré : {output_file}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    
    file_path = args.file_path
    output_file = args.output_file
    
    generate_coordinates(file_path, output_file)