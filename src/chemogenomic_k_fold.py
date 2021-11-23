# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""

from sklearn.utils import shuffle
import pandas as pd
import numpy as np


def chemogenomic_k_fold(k_folds,prot_sequences,smiles_strings,kd_values,prot_sim,smiles_sim):
    kd_pos = shuffle(kd_values.loc[kd_values>5.0])
    kd_neg = shuffle(kd_values.loc[kd_values==5.0])
    
    
    data_split = []
    data_split_flag = []
    
    for i in range(k_folds):
        data_split.append([kd_pos.index[i]])
        data_split_flag.append(True)
        
    # Kd Values > 5.0 (Pos)
    
    for i in range(k_folds,len(kd_pos)):
        if data_split_flag.count(True) == 1:
            position = np.where(np.array(data_split_flag) == True)[0][0]
            data_split[position].extend([kd_pos.index[i]])
            data_split_flag[position] = False
            
        else:
            position = eval_sim(kd_pos.index[i],data_split,data_split_flag,prot_sequences,smiles_strings,
                                prot_sim,smiles_sim,0.5,0.5)
            
            data_split[position].extend([kd_pos.index[i]])
            data_split_flag[position] = False
            
        # Reset Flags    
        if data_split_flag.count(True) == 0:
            data_split_flag = [True]*k_folds
            
            
    # Kd Values == 5.0 (Neg)
    
    for i in range(k_folds):
        data_split[i].extend([kd_neg.index[i]])
    
    # Reset Flags
    data_split_flag = [True]*k_folds
    
    for i in range(k_folds,len(kd_neg)):
        
        if data_split_flag.count(True)==1:
            position = np.where(np.array(data_split_flag) == True)[0][0]
            data_split[position].extend([kd_neg.index[i]])
            data_split_flag[position] = False
            
        else:
            position = eval_sim(kd_neg.index[i],data_split,data_split_flag,prot_sequences,smiles_strings,
                            prot_sim,smiles_sim,0.5,0.5)
            
            data_split[position].extend([kd_neg.index[i]])
            data_split_flag[position] = False
            
        if data_split_flag.count(True) == 0:
            data_split_flag=[True]*k_folds
            
            
    return data_split,kd_pos,kd_neg
            
            



def eval_sim(index,data_split, data_split_flag, prot_sequences, smiles_strings,
             prot_sim, smiles_sim, prot_weight, smiles_weight):
    
    
    protein_sequence = prot_sequences.iloc[index]
    smiles_string = smiles_strings.iloc[index]
    
    similarity_score = []
    
    ## Similarity Values for all comparisons in terms of the median value
    for i in range(len(data_split_flag)):
        if data_split_flag[i]==True:
            
            protein_similarity_list=[]
            smiles_similarity_list=[]
            
            for j in range(len(data_split[i])):
                
                protein_sequence_compare=prot_sequences.iloc[data_split[i][j]]

                smiles_string_compare=smiles_strings.iloc[data_split[i][j]]


                smiles_similarity_list.append(smiles_sim.loc[smiles_sim.index == smiles_string,
                                                             smiles_sim.index == smiles_string_compare].values[0][0])
                
                
                protein_similarity_list.append(prot_sim.loc[prot_sim.index == protein_sequence,
                                                            prot_sim.index == protein_sequence_compare].values[0][0])

              
            protein_similarity_score=np.median(protein_similarity_list)

            smiles_similarity_score=np.median(smiles_similarity_list)

            
            similarity_score.append([protein_similarity_score,smiles_similarity_score])


    ## Cluster Choice
    
    similarity_score_normalized=[i[0]*prot_weight+i[1]*smiles_weight for i in similarity_score]
    
    min_similarity=np.min(similarity_score_normalized)
    
    
    # In the case of equal values
    
    min_index=np.random.choice(np.where(np.array(similarity_score_normalized)==min_similarity)[0],1,replace=False)[0]
    
    
    true_positions=np.where(np.array(data_split_flag)==True)[0]

    return int(true_positions[min_index])
            



if __name__=='__main__':
    data_davis=pd.read_csv('../data/davis/dataset/davis_dataset_processed.csv',sep=',')
    kd_values=data_davis['Kd']
    
    
    prot_sw_score_norm = pd.read_csv('../data/davis/similarity/protein_sw_score_norm.csv',sep=',', header=None, index_col=0)
    
    smiles_ecfp6_sim = pd.read_csv('../data/davis/similarity/smiles_ecfp6_tanimoto_sim.csv',sep=',', header=None, index_col=0)
    

    data_clusters,kd_values_positive,kd_values_negative = chemogenomic_k_fold(6,data_davis['Sequence'],
                                                                              data_davis['SMILES'],kd_values,
                                                                              prot_sw_score_norm,smiles_ecfp6_sim)
    
    
    # for i in range(len(data_clusters)):
    #     if i == len(data_clusters) - 1:
    #         shuffle(pd.DataFrame(data_clusters[i])).to_csv('../data/davis/clusters/test_cluster.csv',index=False,header=False)
    #     else:
    #         shuffle(pd.DataFrame(data_clusters[i])).to_csv('../data/davis/clusters/train_cluster_'+str(i)+'.csv',index=False,header=False)
