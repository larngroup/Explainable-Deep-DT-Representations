# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""

from gradram import *
from gradram_eval_util import *
from dataset_builder_util import *
from cnn_fcnn_model import *
import itertools
import glob
from plots_util import *
      

def get_matching_values(ram,window,motifs):
    
    values = [[matching_eval(matching_spot_window(i[0][0],k,i[1]))
                          for i in list(itertools.product(ram,window))] for k in motifs]
    
    
    
    values_proc = [pd.concat([pd.DataFrame([i[-1] for i in values[k]]).
                                        iloc[pd.Series(window)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(values[k]),len(window))],axis=1) 
                                      for k in range(len(values))]
    
    
    for i in range(len(values_proc)):
        values_proc[i].index = [j for j in window]
        values_proc[i].columns = ['-'.join(j) for j in 
                                      list(itertools.product(['GMP','GAP'],['G','NG']))]
    
    
    
    return values_proc



def get_feature_rel(ram,window,thresholds,sites):
    values = [[[feature_rel_eval(matching_spot_window(i[0][0],k,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram,window))] for j in thresholds] 
                                      for k in sites]
    
    
    
    
    values_proc = [[pd.concat([pd.DataFrame([i[-1] for i in values[m][k]]).
                                        iloc[pd.Series(window)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(values[m][k]),len(window))],axis=1) 
                                            for k in range(len(values[m]))] 
                                            for m in range(len(values))] 
    
    for i in range(len(values_proc)):
        for j in range(len(values_proc[i])):
            values_proc[i][j].index = [k for k in window]
            values_proc[i][j].columns = ['-'.join(k) for k in 
                                      list(itertools.product(['GMP','GAP'],['G','NG']))]
            
    
    
        
    return values_proc


if __name__ == '__main__':
    
    # Davis Dataset
    
    data_path={'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')}
    

    
    protein_data, smiles_data, kd_values = dataset_builder(data_path).transform_dataset('Sequence',
                                                                                        'SMILES',
                                                                                        'Kd',
                                                                                        1400,
                                                                                        72)
    data, prot_dict, smiles_dict, clusters = dataset_builder(data_path).get_data()
    
    davis_prot_seq = data['Sequence']
    davis_smiles_string = data['SMILES']
    
    
    # CNN FCNN Model
    
    cnn_model = tf.keras.models.load_model('../model/cnn_fcnn_model_pad_same',
                                            custom_objects={'c_index':c_index})  
    
    cnn_model.trainable = False
    
    # CNN FCNN Model Last Conv Layers
    
    protein_layer = 'Prot_CNN_2'
    smiles_layer = 'SMILES_CNN_2'
    


    # Matching Pairs
    matching_pairs = json.load(open('../data/davis/binding/davis_scpdb_binding.txt'))
    binding_sites_match = [[int(k) for k in j['Binding Site']] for i,j in matching_pairs.items()]
    
    
    protein_data_match = dataset_builder(data_path).data_conversion([j['Sequence'] 
                                                                for i,j in matching_pairs.items()],
                                                                prot_dict,1400)
    
    
    smiles_data_match = dataset_builder(data_path).data_conversion([j['SMILES'] 
                                                                for i,j in matching_pairs.items()],
                                                              smiles_dict,72)
    
    matching_seq = [j['Sequence'] for i,j in matching_pairs.items()]
    
    
    davis_scpdb_pssm = [pd.read_csv('../data/davis/pssm/pssm_'+str(i)+'.csv',index_col=0) 
                        for i in range(len(matching_pairs))]
    

    
    
    # SC-PDB Pairs
    
    scpdb_binding = json.load(open('../data/scpdb/binding/scpdb_binding.txt'))
    scpdb_binding_sites = [[int(k) for k in j['Binding Site']] for i,j in scpdb_binding.items()]
    
    scpdb_protein_data = dataset_builder(data_path).data_conversion([j['Sequence'] 
                                                                for i,j in scpdb_binding.items()],
                                                                prot_dict,1400)
    
    scpdb_smiles_data = dataset_builder(data_path).data_conversion([j['SMILES'] 
                                                                for i,j in scpdb_binding.items()],
                                                              smiles_dict,72)
    
    scpdb_sequences = [j['Sequence'] for i,j in scpdb_binding.items()]
    
    
    scpdb_pssm = [pd.read_csv('../data/scpdb/pssm/pssm_'+str(i)+'.csv',index_col=0) 
                  for i in range(len(scpdb_binding))]
    


    # GradRAM
    
    grad_eval_model = grad_ram(cnn_model, protein_layer, smiles_layer)
    
    gmp_gap = ['gmp', 'gap']
    guided = [True, False]
    
    ram_match = [calculate_ram(grad_eval_model,protein_data_match,smiles_data_match,i[0],i[1])
                  for i in list(itertools.product(gmp_gap,guided))]
    
    
    ram_scpdb = [calculate_ram(grad_eval_model,scpdb_protein_data,scpdb_smiles_data,i[0],i[1])
                  for i in list(itertools.product(gmp_gap,guided))]
    
    
    
    # PSSM Motifs Eval
    pssm_threshold = [5,6,7,8,9,10] 
    
    
    davis_scpdb_pssm_motifs = [[detect_motif(i,j) for i in davis_scpdb_pssm] for j in pssm_threshold]
    
    davis_scpdb_pssm_motifs_out_bind= [[[m for m in i if (m<j[0] or m>j[-1])] 
                                        for i,j in zip(k,binding_sites_match)] 
                                        for k in davis_scpdb_pssm_motifs]
    
    scpdb_pssm_motifs = [[detect_motif(i,j) for i in scpdb_pssm] for j in pssm_threshold]
    
    scpdb_pssm_motifs_out_bind= [[[m for m in i if (m<j[0] or m>j[-1])] for i,j in zip(k,scpdb_binding_sites)] 
                                  for k in scpdb_pssm_motifs]
    

    # PSSM Motifs - L Grad-RAM Matching
    window_size = [0,1,2,3,4,5]
    
    
    davis_scpdb_motifs_match = get_matching_values(ram_match,window_size,davis_scpdb_pssm_motifs)
    davis_scpdb_motifs_out_match = get_matching_values(ram_match,window_size,davis_scpdb_pssm_motifs_out_bind)
    
    scpdb_motifs_match = get_matching_values(ram_scpdb,window_size,scpdb_pssm_motifs)
    scpdb_motifs_out_match = get_matching_values(ram_scpdb,window_size,scpdb_pssm_motifs_out_bind)
    

    
    # PSSM Motifs - L Grad-RAM Features Relevance
    threshold_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
                                                       
    davis_scpdb_motifs_feature_rel = get_feature_rel(ram_match,window_size,threshold_values,davis_scpdb_pssm_motifs)
    davis_scpdb_motifs_out_feature_rel = get_feature_rel(ram_match,window_size,threshold_values,
                                                         davis_scpdb_pssm_motifs_out_bind)
    
    scpdb_motifs_feature_rel = get_feature_rel(ram_scpdb,window_size,threshold_values,scpdb_pssm_motifs)
    scpdb_motifs_out_feature_rel = get_feature_rel(ram_scpdb,window_size,threshold_values,scpdb_pssm_motifs_out_bind)
    
    



    # # Plots
    pssm_window_heatmap(davis_scpdb_motifs_match, window_size, pssm_threshold, 'Davis ∩ sc-PDB PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(davis_scpdb_motifs_out_match, window_size, pssm_threshold,
                        'Davis ∩ sc-PDB ∉ Binding Region PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(scpdb_motifs_match, window_size, pssm_threshold, 'sc-PDB PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(scpdb_motifs_out_match, window_size, pssm_threshold, 'sc-PDB ∉ Binding Region PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    
    
    
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in davis_scpdb_motifs_feature_rel],
                                    [i*100 for i in threshold_values], pssm_threshold,
                                    'Davis ∩ sc-PDB PSSM Feature Relevance', 'Feature Relevance Threshold',
                                    'PSSM Threshold', False, '')
    
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in davis_scpdb_motifs_out_feature_rel],
                                    [i*100 for i in threshold_values], pssm_threshold,
                                    'Davis ∩ sc-PDB ∉ Binding Region PSSM Feature Relevance',
                                    'Feature Relevance Threshold', 'PSSM Threshold', False, '')
        
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in scpdb_motifs_feature_rel],
                                    [i*100 for i in threshold_values], pssm_threshold,
                                    'sc-PDB PSSM Feature Relevance',
                                    'Feature Relevance Threshold', 'PSSM Threshold', False, '')
            
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in scpdb_motifs_out_feature_rel],
                                    [i*100 for i in threshold_values], pssm_threshold,
                                    'sc-PDB ∉ Binding Region PSSM Feature Relevance',
                                    'Feature Relevance Threshold', 'PSSM Threshold', False, '')
    
    
    
    
    
    
    
    
    


    

  
    