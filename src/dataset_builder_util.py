# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import re
from operator import itemgetter

# Provide a dictionary data path : {'data' : path, 'prot_dic' : path, 'smiles_dic' : path, 'clusters' : [paths]}
# Davis Dataset: [29187,10]: 
# 0	Acession Number
# 1	Gene
# 2	Kinase
# 3	Sequence
# 4	Compound
# 5	PubChem_Cid
# 6	SMILES
# 7	Kd

# Protein Dictionary [20] : 'Amino Acid'  : Number
# SMILES Dictionary  [26] : 'SMILES Char' : Number
# Clusters 0 - Train - [4863], 1 - Train - [4863], 2 - Train - [4863], 3 - Train - [4863], 4 - Train - [4863], Test - [4866]

class dataset_builder():
    def __init__(self,data_path,**kwargs):
        super(dataset_builder,self).__init__(**kwargs)
        self.data_path = data_path
        
        
    def get_data(self):
        dataset = pd.read_csv(self.data_path['data'], sep=',', memory_map=True)
        prot_dictionary = json.load(open(self.data_path['prot_dic']))
        smiles_dictionary = json.load(open(self.data_path['smiles_dic']))
        clusters = []
        for i in self.data_path['clusters']:
            if 'test' in i:
                clusters.append(('test', pd.read_csv(i,header=None)))
            else:
                clusters.append(('train', pd.read_csv(i,header=None)))

        
        return (dataset, prot_dictionary, smiles_dictionary, clusters)
        
    def data_conversion(self,data,dictionary,max_len):
        keys=list(i for i in dictionary.keys() if len(i)>1)

        if len(keys)==0:
            data=pd.DataFrame([list(i) for i in data])

        else:
            char_list=[]
            for i in data:
                positions=[]
                for j in keys:
                    positions.extend([(k.start(),k.end()-k.start()) for k in re.finditer(j,i)])
                    
                positions=sorted(positions,key=itemgetter(0))
                
                if len(positions)==0:
                    char_list.append(list(i))

                else:
                    new_list=[]
                    j=0
                    positions_start=[k[0] for k in positions]
                    positions_len=[k[1] for k in positions]
                    
                    while j<len(i):
                        if j in positions_start:
                            new_list.append(str(i[j]+i[j+positions_len[positions_start.index(j)]-1]))
                            j=j+positions_len[positions_start.index(j)]
                        else:
                            new_list.append(i[j])
                            j=j+1
                    char_list.append(new_list)

            data=pd.DataFrame(char_list)

                    
        data.replace(dictionary,inplace=True)

        data = data.fillna(0)
        if len(data.iloc[0,:]) == max_len:
            return data
        else:
            zeros_array = np.zeros(shape=(len(data.iloc[:,0]),max_len-len(data.iloc[0,:])))
            data = pd.concat((data,pd.DataFrame(zeros_array)),axis=1)
            return data
    
    
    def transform_dataset(self, protein_column, smiles_column, kd_column, prot_max_len, smiles_max_len):
        protein_data = self.data_conversion(self.get_data()[0][protein_column], 
                                            self.get_data()[1],prot_max_len).astype('int64')
        
        smiles_data  = self.data_conversion(self.get_data()[0][smiles_column], 
                                            self.get_data()[2],smiles_max_len).astype('int64')
        
        
        kd_values = self.get_data()[0][kd_column].astype('float64')
        
        return (tf.convert_to_tensor(protein_data, dtype=tf.int64), tf.convert_to_tensor(smiles_data, dtype=tf.int64),
                tf.convert_to_tensor(kd_values, dtype=tf.float32))
        
 
            
        
        
        
        