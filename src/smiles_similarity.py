# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""


from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np


# Requires RDKIT Env
# SMILES Similarity Options: RDKIT Similarity Function, Morgan Radius 2 (Similar to ECFP4) and Morgan Radius 3 (Similar to ECFP6)
def smiles_similarity(smiles_a,smiles_b,fps_option):
    smiles_a_mol = Chem.MolFromSmiles(smiles_a)
    smiles_b_mol = Chem.MolFromSmiles(smiles_b)
    
    
    if fps_option=="rdkit_fingerprint":
        smiles_a_fp = Chem.RDKFingerprint(smiles_a_mol)
        smiles_b_fp = Chem.RDKFingerprint(smiles_b_mol)
        
    elif fps_option=="morgan_2_ECFP4":
        smiles_a_fp = AllChem.GetMorganFingerprintAsBitVect(smiles_a_mol,2,nBits=2048)
        smiles_b_fp = AllChem.GetMorganFingerprintAsBitVect(smiles_b_mol,2,nBits=2048)

    elif fps_option=="morgan_3_ECFP6":
        smiles_a_fp = AllChem.GetMorganFingerprintAsBitVect(smiles_a_mol,3,nBits=2048)
        smiles_b_fp = AllChem.GetMorganFingerprintAsBitVect(smiles_b_mol,3,nBits=2048)
        
    similarity = DataStructs.FingerprintSimilarity(smiles_a_fp,smiles_b_fp,metric=DataStructs.TanimotoSimilarity)
    
    return similarity


if __name__=='__main__':
    data_davis=pd.read_csv('../data/davis/dataset/davis_dataset_processed.csv',sep=',',memory_map=True)
    smiles=data_davis['SMILES']
    
    smiles_unique=smiles.unique()
    
    similarity_matrix = np.zeros(shape=(len(smiles_unique),len(smiles_unique)))
    for i in range(len(smiles_unique)):
        for j in range(len(smiles_unique)):
            similarity_matrix[i,j] = smiles_similarity(smiles_unique[i],smiles_unique[j],'morgan_3_ECFP6')
            
    similarity_matrix = pd.DataFrame(similarity_matrix)
    similarity_matrix.index = smiles_unique
    # similarity_matrix.to_csv('../data/davis/similarity/smiles_ecfp6_tanimoto_sim.csv',header=None,index=True)

    
    
    
     
