# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""

import tensorflow as tf

class grad_ram:
    def __init__(self,model,layer_protein,layer_smiles,**kwargs):

        self.model = model
        self.layer_protein = layer_protein
        self.layer_smiles = layer_smiles
        
        
    def calculate_gradram(self, protein_input, smiles_input, gmp_gap, guided):
        prot_conv_layer = self.model.get_layer(self.layer_protein)
        smiles_conv_layer = self.model.get_layer(self.layer_smiles)
        
        # Shape : (Batch,1400,128), (Batch,72,128), (Batch,1)
        gradram_model = tf.keras.Model(inputs=self.model.inputs, outputs=[prot_conv_layer.output,smiles_conv_layer.output,
                                                                          self.model.outputs])
        
        with tf.GradientTape(persistent=True) as tape:
            (conv_out_prot, conv_out_smiles, model_out) = gradram_model([protein_input,smiles_input])
            
        grads_prot = tape.gradient(model_out, conv_out_prot)  # (Batch,1400,128)
        grads_smiles = tape.gradient(model_out, conv_out_smiles) # (Batch,72,128)
        
        del tape

            
        if guided: # Gradients > 0 
                    
            grads_prot = tf.cast(conv_out_prot > 0, "float32") * tf.cast(grads_prot > 0, "float32") * grads_prot
            grads_smiles = tf.cast(conv_out_smiles > 0, "float32") * tf.cast(grads_smiles > 0, "float32") * grads_smiles
        
        
        if gmp_gap == 'gmp':   # Global Max Pooling
                        
            weights_prot = tf.reduce_max(grads_prot,axis=1) # (Batch, 128)
            weights_smiles = tf.reduce_max(grads_smiles, axis=1) # (Batch, 128)
            
        if gmp_gap == 'gap':   # Global Average Pooling
            
            weights_prot = tf.reduce_mean(grads_prot,axis=1) # (Batch, 128)
            weights_smiles = tf.reduce_mean(grads_smiles, axis=1) # (Batch, 128)

        
        
        ram_prot = tf.nn.relu(tf.reduce_sum(tf.multiply(tf.transpose(conv_out_prot,perm=[0,2,1]),
                                                           tf.stack([weights_prot],axis=2)),axis=1)) # (Batch, 1400)
        
        ram_smiles = tf.nn.relu(tf.reduce_sum(tf.multiply(tf.transpose(conv_out_smiles,perm=[0,2,1]),
                                                           tf.stack([weights_smiles],axis=2)),axis=1)) # (Batch,72)
        
        
        return ram_prot, ram_smiles
        

        


    
        
        



    
    
