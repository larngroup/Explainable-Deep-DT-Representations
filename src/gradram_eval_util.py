# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""
import numpy as np

def calculate_ram(gradram_model, proteins, smiles, gmp_gap, guided):
    ram_protein, ram_smiles = gradram_model.calculate_gradram(proteins, smiles, gmp_gap, guided)
        
    return ram_protein, ram_smiles
    

def matching_spot_window(ram_protein, binding_locations, window_size=0):
    match = []
    for i in range(len(ram_protein)):
        binding_info={}
        if len(binding_locations[i])==0:
            match.append(0)
            continue
        else:
            for j in range(len(binding_locations[i])):
                if len(binding_locations[i]) == 1:
                    min_index = np.clip(binding_locations[i][j] - window_size, 0, binding_locations[i][j])
                    max_index = np.clip(binding_locations[i][j] + window_size, binding_locations[i][j], len(ram_protein[i])-1)
                    
                    if min_index == max_index:
                        window_region = ram_protein[i,min_index]
                    else:
                        window_region = ram_protein[i,min_index:max_index+1]
                    
                elif j == 0 :
                    min_index = np.clip(binding_locations[i][j] - window_size, 0, binding_locations[i][j])
                    
                    max_index = np.clip(binding_locations[i][j] + window_size, binding_locations[i][j], binding_locations[i][j+1]-1)

                    if min_index == max_index:
                        window_region = ram_protein[i,min_index]
                    else:
                        window_region = ram_protein[i,min_index:max_index+1]

                elif j == len(binding_locations[i])-1:
                    
                    min_index = np.clip(binding_locations[i][j] - window_size, binding_locations[i][j-1]+1, binding_locations[i][j])

                    max_index = np.clip(binding_locations[i][j] + window_size, binding_locations[i][j], len(ram_protein[i])-1)

                    if min_index == max_index:
                        window_region = ram_protein[i,min_index]
                    else:
                        window_region = ram_protein[i,min_index:max_index+1]

                else:
                    min_index = np.clip(binding_locations[i][j] - window_size, binding_locations[i][j-1]+1, binding_locations[i][j])

                    max_index = np.clip(binding_locations[i][j] + window_size, binding_locations[i][j], binding_locations[i][j+1]-1)

                    if min_index == max_index:
                        window_region = ram_protein[i,min_index]
                    else:
                        window_region = ram_protein[i,min_index:max_index+1]


                if window_region.shape == [] : 
                    window_region_pos = [1 if window_region.numpy()>0 else 0] 
                    
                else:
                    window_region_pos = []
                    for k in window_region.numpy():
                        if k >0 :
                            window_region_pos.append(1)
                        else:
                            window_region_pos.append(0)

                binding_info[j] = {'Seq Bind Pos' : binding_locations[i][j], 'Window Len': window_size,
                                   'Effective Window Len': window_region.numpy().size,
                                   'Window Pos': [min_index,max_index] ,
                                   'Match Count': window_region_pos.count(1),
                                   'Window Region': window_region.numpy()}
            match.append(binding_info)
            
    return match
    


def matching_eval(match_window):
    num_spots = []
    mean_match = []
    
    for i in match_window:
        if i == 0 :
            continue
        else : 
            num_bind_spot = len(i)
            num_spots.append(num_bind_spot)
            count_match = sum([1 for k,v in i.items() if v['Match Count'] != 0])
            mean_match.append(count_match/num_bind_spot)

    

    average_values = []
    for i in range(len(num_spots)):
        avg = num_spots[i]/(sum(num_spots)) * mean_match[i]
        average_values.append(avg)
    
    average_values = sum(average_values) * 100
    
    
    
    return [mean_match,num_spots,average_values]
    



def feature_rel_eval(match_window, ram_prot, threshold):
    pos_match = []
    num_window_pos = []


    for i in range(len(match_window)):
        if match_window[i] == 0 :
            continue
        else:
            ram_prot_pos = sorted(ram_prot[i,:][ram_prot[i,:]>0],reverse=True)[:int(threshold*len(ram_prot[i,:][ram_prot[i,:]>0]))]

            window_values = np.hstack([v['Window Region'] for k,v in match_window[i].items()])
            window_values_pos = window_values[window_values>0]
        
            num_window_pos.append(len(window_values_pos))
            
            if len(window_values_pos)==0:
                pos_match.append(0)
            else:
                count_match = [1 for j in window_values_pos if j in ram_prot_pos].count(1)

                pos_match.append(count_match/len(window_values_pos))

        
    average_values = []
    for i in range(len(num_window_pos)):
        avg = num_window_pos[i]/(sum(num_window_pos)) * pos_match[i]
        average_values.append(avg)
    
    average_values = sum(average_values) * 100
    

    return [num_window_pos, pos_match,average_values]


def detect_motif(pssm,threshold):
    idx_motif = []
    for i in range(len(pssm)):
        if int(pssm.iloc[i,pssm.columns.get_loc(pssm.index[i])])>=threshold:

            idx_motif.append(i)

    return idx_motif
           