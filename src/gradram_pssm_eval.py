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
    data,prot_dict,smiles_dict,clusters = dataset_builder(data_path).get_data()
    
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
    
    binding_sites_match = [sorted(list(np.unique(j['Binding Site']).astype(np.int32)-1)) 
                           for i,j in matching_pairs.items()]
    
    
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
    
    scpdb_binding_sites = [sorted(list(np.unique(j['Binding Sites']).astype(np.int32)-1)) 
                           for i,j in scpdb_binding.items()]
    
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
    
    window_size = [0,1,2,3,4,5]
    threshold_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    pssm_threshold = [5,6,7,8,9,10] 
    
    
    davis_scpdb_pssm_motifs = [[detect_motif(i,j) for i in davis_scpdb_pssm] for j in pssm_threshold]
    
    davis_scpdb_pssm_motifs_out_bind= [[[m for m in i if (m<j[0] or m>j[-1])] 
                                        for i,j in zip(k,binding_sites_match)] 
                                       for k in davis_scpdb_pssm_motifs]
    
    scpdb_pssm_motifs = [[detect_motif(i,j) for i in scpdb_pssm] for j in pssm_threshold]
    
    scpdb_pssm_motifs_out_bind= [[[m for m in i if (m<j[0] or m>j[-1])] for i,j in zip(k,scpdb_binding_sites)] 
                                 for k in scpdb_pssm_motifs]
    

    # PSSM Motifs - L Grad-RAM Matching
    
    davis_scpdb_motifs_match = [[matching_eval(matching_spot_window(i[0][0],k,i[1]))
                          for i in list(itertools.product(ram_match,window_size))] for k in davis_scpdb_pssm_motifs]
    
    
    davis_scpdb_motifs_out_match = [[matching_eval(matching_spot_window(i[0][0],k,i[1]))
                          for i in list(itertools.product(ram_match,window_size))] for k in davis_scpdb_pssm_motifs_out_bind]                                    




    scpdb_motifs_match = [[matching_eval(matching_spot_window(i[0][0],k,i[1]))
                          for i in list(itertools.product(ram_scpdb,window_size))] for k in scpdb_pssm_motifs]    
    
    
    scpdb_motifs_out_match = [[matching_eval(matching_spot_window(i[0][0],k,i[1]))
                          for i in list(itertools.product(ram_scpdb,window_size))] for k in scpdb_pssm_motifs_out_bind] 



    
        
    davis_scpdb_motifs_match_proc = [pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_motifs_match[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_motifs_match[k]),6)],axis=1) 
                                     for k in range(len(davis_scpdb_motifs_match))]
    
    davis_scpdb_motifs_out_match_proc = [pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_motifs_out_match[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_motifs_out_match[k]),6)],axis=1) 
                                        for k in range(len(davis_scpdb_motifs_out_match))]
    
    
    scpdb_motifs_match_proc = [pd.concat([pd.DataFrame([i[-1] for i in scpdb_motifs_match[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_motifs_match[k]),6)],axis=1) 
                               for k in range(len(scpdb_motifs_match))]    
    
    
    scpdb_motifs_out_match_proc = [pd.concat([pd.DataFrame([i[-1] for i in scpdb_motifs_out_match[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_motifs_out_match[k]),6)],axis=1) 
                                   for k in range(len(scpdb_motifs_out_match))] 
    
    
    
    
    # PSSM Motifs - L Grad-RAM Features Relevance
    
    davis_scpdb_motifs_feature_rel = [[[feature_rel_eval(matching_spot_window(i[0][0],k,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_match,window_size))] for j in threshold_values] 
                                      for k in davis_scpdb_pssm_motifs]


    davis_scpdb_motifs_out_feature_rel = [[[feature_rel_eval(matching_spot_window(i[0][0],k,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_match,window_size))] for j in threshold_values] 
                                          for k in davis_scpdb_pssm_motifs_out_bind]




    scpdb_motifs_feature_rel = [[[feature_rel_eval(matching_spot_window(i[0][0],k,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_scpdb,window_size))] for j in threshold_values] 
                                for k in scpdb_pssm_motifs]



    scpdb_motifs_out_feature_rel = [[[feature_rel_eval(matching_spot_window(i[0][0],k,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_scpdb,window_size))] for j in threshold_values] 
                                    for k in scpdb_pssm_motifs_out_bind]





    davis_scpdb_motifs_feature_rel_proc = [[pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_motifs_feature_rel[m][k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_motifs_feature_rel[m][k]),6)],axis=1) 
                                            for k in range(len(davis_scpdb_motifs_feature_rel[m]))] 
                                           for m in range(len(davis_scpdb_motifs_feature_rel))] 


    davis_scpdb_motifs_out_feature_rel_proc = [[pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_motifs_out_feature_rel[m][k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_motifs_out_feature_rel[m][k]),6)],axis=1) 
                                                for k in range(len(davis_scpdb_motifs_out_feature_rel[m]))] 
                                               for m in range(len(davis_scpdb_motifs_out_feature_rel))]     
    



    scpdb_motifs_feature_rel_proc = [[pd.concat([pd.DataFrame([i[-1] for i in scpdb_motifs_feature_rel[m][k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_motifs_feature_rel[m][k]),6)],axis=1) 
                                      for k in range(len(scpdb_motifs_feature_rel[m]))] 
                                     for m in range(len(scpdb_motifs_feature_rel))]   
    
    
    scpdb_motifs_out_feature_rel_proc = [[pd.concat([pd.DataFrame([i[-1] for i in scpdb_motifs_out_feature_rel[m][k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_motifs_out_feature_rel[m][k]),6)],axis=1) 
                                          for k in range(len(scpdb_motifs_out_feature_rel[m]))] 
                                         for m in range(len(scpdb_motifs_out_feature_rel))] 


    # Plots
    pssm_window_heatmap(davis_scpdb_motifs_match_proc, window_size, pssm_threshold, 'Davis ∩ sc-PDB PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(davis_scpdb_motifs_out_match_proc, window_size, pssm_threshold,
                        'Davis ∩ sc-PDB ∉ Binding Region PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(scpdb_motifs_match_proc, window_size, pssm_threshold, 'sc-PDB PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    pssm_window_heatmap(scpdb_motifs_out_match_proc, window_size, pssm_threshold, 'sc-PDB ∉ Binding Region PSSM Matching',
                        'Window Length', 'PSSM Threshold', False, '')
    
    
    
    
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in davis_scpdb_motifs_feature_rel_proc],
                                   [10,20,30,40,50,60,70], pssm_threshold,
                                   'Davis ∩ sc-PDB PSSM Feature Relevance', 'Feature Relevance Threshold',
                                   'PSSM Threshold', False, '')
    
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in davis_scpdb_motifs_out_feature_rel_proc],
                                   [10,20,30,40,50,60,70], pssm_threshold,
                                   'Davis ∩ sc-PDB ∉ Binding Region PSSM Feature Relevance',
                                   'Feature Relevance Threshold', 'PSSM Threshold', False, '')
        
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in scpdb_motifs_feature_rel_proc],
                                   [10,20,30,40,50,60,70], pssm_threshold,
                                   'sc-PDB PSSM Feature Relevance',
                                   'Feature Relevance Threshold', 'PSSM Threshold', False, '')
            
    pssm_feature_relevance_heatmap([[np.mean(i.iloc[:,0]) for i in j] for j in scpdb_motifs_out_feature_rel_proc],
                                   [10,20,30,40,50,60,70], pssm_threshold,
                                   'sc-PDB ∉ Binding Region PSSM Feature Relevance',
                                   'Feature Relevance Threshold', 'PSSM Threshold', False, '')
    
    
    
    
    
    
    
    
    


    

  
    