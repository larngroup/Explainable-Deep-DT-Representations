# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""

from rdkit import Chem

# RDKIT Canonical SMILES

def rdkit_canonical(smiles):
    mol=Chem.MolFromSmiles(smiles)
    can_rdkit_smiles=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
    return can_rdkit_smiles
