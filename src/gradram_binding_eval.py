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
    binding_sites_match = [sorted(list(np.unique(j['Binding Site']).astype(np.int32)-1)) for i,j in matching_pairs.items()]
    
    
    protein_data_match = dataset_builder(data_path).data_conversion([j['Sequence'] 
                                                                for i,j in matching_pairs.items()],
                                                                prot_dict,1400)
    
    
    smiles_data_match = dataset_builder(data_path).data_conversion([j['SMILES'] 
                                                                for i,j in matching_pairs.items()],
                                                              smiles_dict,72)
    
    matching_seq = [j['Sequence'] for i,j in matching_pairs.items()]
    
    
    # SC-PDB Pairs
    
    scpdb_binding = json.load(open('../data/scpdb/binding/scpdb_binding.txt'))
    scpdb_binding_sites = [sorted(list(np.unique(j['Binding Sites']).astype(np.int32)-1)) for i,j in scpdb_binding.items()]
    
    scpdb_protein_data = dataset_builder(data_path).data_conversion([j['Sequence'] 
                                                                for i,j in scpdb_binding.items()],
                                                                prot_dict,1400)
    
    scpdb_smiles_data = dataset_builder(data_path).data_conversion([j['SMILES'] 
                                                                for i,j in scpdb_binding.items()],
                                                              smiles_dict,72)
    
    scpdb_sequences = [j['Sequence'] for i,j in scpdb_binding.items()]
    
    


    # GradRAM
    
    grad_eval_model = grad_ram(cnn_model, protein_layer, smiles_layer)
    
    gmp_gap = ['gmp', 'gap']
    guided = [True, False]
    
    ram_match = [calculate_ram(grad_eval_model,protein_data_match,smiles_data_match,i[0],i[1])
                  for i in list(itertools.product(gmp_gap,guided))]
    
    
    ram_scpdb = [calculate_ram(grad_eval_model,scpdb_protein_data,scpdb_smiles_data,i[0],i[1])
                  for i in list(itertools.product(gmp_gap,guided))]



    # GradRAM Eval   
    
    window_size = [0,1,2,3,4,5]
    threshold_values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    
    # Binding Spots - L Grad-RAM Matching
    
    davis_scpdb_binding_match = [matching_eval(matching_spot_window(i[0][0],binding_sites_match,i[1]))
                          for i in list(itertools.product(ram_match,window_size))]
    
    davis_scpdb_binding_match_proc = pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_binding_match]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_binding_match),6)],axis=1)
    

    davis_scpdb_binding_match_proc.index = [j for j in window_size]
    davis_scpdb_binding_match_proc.columns = ['-'.join(i) for i in 
                                      list(itertools.product(gmp_gap,['G','NG']))]      
    
    
    
                                          
    scpdb_binding_match = [matching_eval(matching_spot_window(i[0][0],scpdb_binding_sites,i[1]))
                          for i in list(itertools.product(ram_scpdb,window_size))]
    
    
    scpdb_binding_match_proc = pd.concat([pd.DataFrame([i[-1] for i in scpdb_binding_match]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_binding_match),6)],axis=1)
    
    
    scpdb_binding_match_proc.index = [j for j in window_size]
    scpdb_binding_match_proc.columns = ['-'.join(i) for i in 
                                      list(itertools.product(gmp_gap,['G','NG']))]  


                         
   
    # Binding Spots - L Grad-RAM Features Relevance
    
    
    davis_scpdb_bind_feature_rel = [[feature_rel_eval(matching_spot_window(i[0][0],binding_sites_match,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_match,window_size))] for j in threshold_values]
    
    
    
    davis_scpdb_bind_feature_rel_proc = [pd.concat([pd.DataFrame([i[-1] for i in davis_scpdb_bind_feature_rel[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(davis_scpdb_bind_feature_rel[k]),6)],axis=1) for k in range(len(davis_scpdb_bind_feature_rel))]
    
    
    for i in range(len(davis_scpdb_bind_feature_rel_proc)):
        davis_scpdb_bind_feature_rel_proc[i].index = [j for j in window_size]
        davis_scpdb_bind_feature_rel_proc[i].columns = ['-'.join(i) for i in 
                                      list(itertools.product(gmp_gap,['G','NG']))]
        
    

    scpdb_binding_feature_rel = [[feature_rel_eval(matching_spot_window(i[0][0],scpdb_binding_sites,i[1]),i[0][0].numpy(),j)
                          for i in list(itertools.product(ram_scpdb,window_size))] for j in threshold_values]
    
    
    
    scpdb_binding_feature_rel_proc = [pd.concat([pd.DataFrame([i[-1] for i in scpdb_binding_feature_rel[k]]).
                                        iloc[pd.Series(window_size)+j,:].reset_index(drop=True) 
                                        for j in range(0,len(scpdb_binding_feature_rel[k]),6)],axis=1) for k in range(len(scpdb_binding_feature_rel))]
    
    

    for i in range(len(scpdb_binding_feature_rel_proc)):
        scpdb_binding_feature_rel_proc[i].index = [j for j in window_size]
        scpdb_binding_feature_rel_proc[i].columns = ['-'.join(i) for i in 
                                      list(itertools.product(gmp_gap,['G','NG']))]
        
        
    # Plots
    feature_rel_density_map(davis_scpdb_bind_feature_rel_proc,threshold_values,window_size,'Davis ∩ sc-PDB Feature Relevance',
                  'Window Length','Feature Relevance Threshold',False,'')
    
    feature_rel_density_map(scpdb_binding_feature_rel_proc,threshold_values,window_size,'sc-PDB Feature Relevance',
                  'Window Length','Feature Relevance Threshold',False,'')
    
    
    sequences = [('davis',matching_seq[i]) for i in [5,14,17,31]] + [('scpdb',scpdb_sequences[i]) for i in [14,145,177,186]]
    ram_values = [ram_match[0][0][i,:] for i in [5,14,17,31]] + [ram_scpdb[0][0][i,:] for i in [14,145,177,186]]
    binding_values = [binding_sites_match[i] for i in [5,14,17,31]] + [scpdb_binding_sites[i] for i in [14,145,177,186]]
    labels = ['ABL1\n(H396P)\nNP*', 'BRAF', 'CDK8', 'RIPK2'] + ['Chk1','SYK','TTBK1','PDPK1']
    colors = ['red','blue'] 
    texts = ["Davis ∩ sc-PDB Binding Sites", "sc-PDB Binding Sites"]
    markers = ['o','o'] 
    xlabel = 'Protein Sequence Length'
    gradram_plot(sequences,ram_values,binding_values,labels,colors,texts,markers,xlabel,False,'')
    
  
    
