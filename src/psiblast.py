# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""

import subprocess

# Standalone Blast 2.11.0
def psiblast_pssm (option,davis_dir,scpdb_dir,sequence_data,blast_db_dir = 'D:/blast-2.11.0+/db/nr'):
    if option == 'davis':
        data_dir = davis_dir
    elif option == 'scpdb':
        data_dir = scpdb_dir
        
    for i in range(len(sequence_data)):
        # Convert Sequence to Fasta
        seq_name = 'seq'+'_'+str(i)+'.fasta'
        open(data_dir + seq_name,'w').write(sequence_data[i])
        
        query = data_dir + seq_name
        pssm_out = data_dir + str(i) + '_ascii_pssm'
        cmd = 'psiblast -query ' + query + ' -db '+blast_db_dir+' -taxids 9606 -num_iterations 3 -num_threads 12 -out_ascii_pssm ' + pssm_out 
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=None, shell=True).communicate()
    
    
    
    
    
    
    
    
    
    
    
    
    