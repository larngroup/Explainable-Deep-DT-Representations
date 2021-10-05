# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""
import tensorflow as tf
from dataset_builder_util import *
import itertools
import pandas as pd
from plots_util import *
import glob
from scipy import stats

def save_func(file_path,values):
    file=[i.rstrip().split(',') for i in open(file_path).readlines()]
    file.append(values)
    file=pd.DataFrame(file)
    file.to_csv(file_path,header=None,index=None)
    
    
def inference_metrics(model,data):
    start = time.time()
    pred_values = model.predict([data[0],data[1]])
    end = time.time()
    inf_time = end-start

    metrics = {'MSE': mse(data[2],pred_values), 'RMSE': mse(data[2],pred_values,squared=False),
               'CI':c_index(data[2],pred_values).numpy(), 'R2': r2s(data[2],pred_values),
               'Spearman':stats.spearmanr(data[2],pred_values)[0],'Time':inf_time}
    
    return metrics


def generate_one_hot_layer(input_dim, input_length):
    def one_hot_enconding(x, num_classes):
        one_hot = tf.one_hot(tf.cast(x, 'uint8'), depth=num_classes)
        one_hot = tf.gather(
            one_hot, [i for i in range(1, num_classes)], axis=2)
        return one_hot

    one_hot_layer = tf.keras.layers.Lambda(one_hot_enconding, arguments={
                                           'num_classes': input_dim}, input_shape=(input_length,))
    return one_hot_layer

def c_index(y_true,y_pred):
    matrix_pred=tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    matrix_pred=tf.cast(matrix_pred == 0.0, tf.float32) * 0.5 + tf.cast(matrix_pred > 0.0, tf.float32)
    
    
    matrix_true=tf.subtract(tf.expand_dims(y_true, -1), y_true)
    matrix_true=tf.cast(matrix_true>0.0,tf.float32)
    
    matrix_true_position=tf.where(tf.equal(matrix_true,1))
    
    matrix_pred_values=tf.gather_nd(matrix_pred,matrix_true_position)
    
    # If equal to zero then it returns zero, else return the result of the division
    result=tf.where(tf.equal(tf.reduce_sum(matrix_pred_values),0),0.0,tf.reduce_sum(matrix_pred_values)/tf.reduce_sum(matrix_true))
    
    return result


class CNN_FCNN_Model:
    def __init__(self, prot_len, smiles_len, type_emb, prot_dict_len, smiles_dict_len, prot_emb_size,
                 smiles_emb_size, num_cnn_layers_prot, cnn_layers_prot_filters,
                 cnn_layers_prot_filter_w, prot_cnn_act_func, prot_padding_opt,
                 num_cnn_layers_smiles, cnn_layers_smiles_filters, cnn_layers_smiles_filter_w,
                 smiles_cnn_act_func, smiles_padding_opt, num_fcnn_layers, fcnn_act_func, fcnn_units,
                 dropout_rate, output_act_func, optimizer_fn, loss_fn, metrics_list, **kwargs):


        self.prot_input = tf.keras.Input(
            shape=(prot_len,), dtype=tf.int64, name='Protein_Input')
        self.smiles_input = tf.keras.Input(
            shape=(smiles_len,), dtype=tf.int64, name='SMILES_Input')

        if type_emb == 'one_hot':
            self.prot_emb = generate_one_hot_layer(
                prot_dict_len+1, prot_len)
            
            self.prot_emb._name='Prot_Emb'
            
            self.smiles_emb = generate_one_hot_layer(
                smiles_dict_len+1, smiles_len)
            
            self.smiles_emb._name='SMILES_Emb'

        elif type_emb == 'word_embedding':
            
            self.prot_emb = tf.keras.layers.Embedding(
                prot_dict_len+1, prot_emb_size, name='Prot_Emb')
            self.smiles_emb = tf.keras.layers.Embedding(
                smiles_dict_len+1, smiles_emb_size, name='SMILES_Emb')

        self.prot_cnn_layers = [tf.keras.layers.Conv1D(filters=cnn_layers_prot_filters[i],
                                                       kernel_size=cnn_layers_prot_filter_w[i],
                                                       strides=1, padding=prot_padding_opt,
                                                       activation = prot_cnn_act_func, name='Prot_CNN_%d'%i)
                                for i in range(num_cnn_layers_prot)]
        
        
        self.smiles_cnn_layers = [tf.keras.layers.Conv1D(filters=cnn_layers_smiles_filters[i],
                                                       kernel_size=cnn_layers_smiles_filter_w[i],
                                                       strides=1, padding = smiles_padding_opt,
                                                       activation = smiles_cnn_act_func, name='SMILES_CNN_%d'%i)
                                for i in range(num_cnn_layers_smiles)]
        
        
        self.prot_global_pool = tf.keras.layers.GlobalMaxPool1D(name ='Prot_Global_Max')
        self.smiles_global_pool = tf.keras.layers.GlobalMaxPool1D(name ='SMILES_Global_Max')
        
        
        self.concatenate_layer = tf.keras.layers.Concatenate(name='Concatenate')
        
        self.fcnn_layers = [tf.keras.layers.Dense(units=fcnn_units[i],activation=fcnn_act_func, name='Dense_%d'%i)
                            for i in range(num_fcnn_layers)]
        
        self.dropout_layers = [tf.keras.layers.Dropout(rate=dropout_rate[i], name='Dropout_%d'%i) for i in range(num_fcnn_layers-1)]
        
        self.output_layer = tf.keras.layers.Dense(units=1, activation = output_act_func, kernel_initializer='normal')
        
        
        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        self.metrics_list = metrics_list
        
        self.model = self.build_model()
           
    def build_model(self):
        
        prot_in = self.prot_input
        smiles_in = self.smiles_input
        
        prot_out = self.prot_emb(prot_in)
        smiles_out = self.smiles_emb(smiles_in)
        
        for p_cnn_layer in self.prot_cnn_layers:
            prot_out = p_cnn_layer(prot_out)
            
        for s_cnn_layer in self.smiles_cnn_layers:
            smiles_out = s_cnn_layer(smiles_out)
            
        prot_out = self.prot_global_pool(prot_out)
        smiles_out = self.smiles_global_pool(smiles_out)
        
        deep_features = self.concatenate_layer([prot_out,smiles_out])
        
        for i in range(len(self.fcnn_layers)-1):
            deep_features = self.dropout_layers[i](self.fcnn_layers[i](deep_features))
            
        deep_features = self.fcnn_layers[-1](deep_features)
        
        out = self.output_layer(deep_features)
    
    
        cnn_fcnn_model = tf.keras.Model(inputs=[prot_in,smiles_in],outputs=out, name='CNN_FCNN_Model')
        
        cnn_fcnn_model.compile(optimizer=self.optimizer_fn,loss=self.loss_fn,metrics=self.metrics_list)
        
        return cnn_fcnn_model
    
    
    def fit_model(self,dataset, batch, epochs, callback_list = None, val_option = False, val_dataset = None):
        #model = self.build_model()
        protein_data, smiles_data, kd_values = dataset

        
        callback_list = callback_list

        
        if not val_option:
            self.model.fit(x=[protein_data,smiles_data], y=kd_values, batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list)
            
        else:
            protein_val_data, smiles_val_data, kd_val_values = val_dataset
            
            self.model.fit(x=[protein_data,smiles_data], y=kd_values, 
                      batch_size = batch, epochs = epochs,
                      verbose = 2, callbacks = callback_list, 
                      validation_data=([protein_val_data, 
                                        smiles_val_data], kd_val_values))
            
        return self.model


def grid_search(parameters,data,k_folds,folder,save_file):
    metrics_results=[[],[],[]]
    for num_run in range(len(k_folds)):
        print("-------------------//--------------------")
        print("Run: "+str(num_run))
        print("-------------------//--------------------")
        index_train=list(itertools.chain.from_iterable([k_folds[i] for i in range(len(k_folds)) if i!=num_run]))

        index_val=k_folds[num_run]

        
        data_train = [tf.gather(i,index_train) for i in data]

        data_val = [tf.gather(i,index_val) for i in data]
        
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_root_mean_squared_error',
                                         min_delta = 0.001, patience = 50, mode = 'min', 
                                         restore_best_weights=True)
        
        
        
        file_name = ['model_'+str(parameters['Num_CNN_Prot'])+'_'+str(parameters['CNN_Prot_Filters'])+'_'+
                     str(parameters['CNN_Prot_Filters_W'])+'_'+str(parameters['Num_CNN_SMILES'])+'_'+
                     str(parameters['CNN_SMILES_Filters'])+'_'+str(parameters['CNN_SMILES_Filters_W'])+'_'+
                     str(parameters['FCNN_Layers'])+'_'+'_'+str(parameters['FCNN_Layers_Units'])+'_'+
                     str(parameters['Dropout'])+'_'+str(num_run)]

        
        mc = tf.keras.callbacks.ModelCheckpoint(filepath = folder+file_name[0], monitor = 'val_root_mean_squared_error',
                                           save_best_only=True, save_weights_only = False, mode = 'min')
        
        callbacks = [es,mc]
        
        cnn_fcnn_model = CNN_FCNN_Model(1400,72,'one_hot',20,26,None,None,parameters['Num_CNN_Prot'],parameters['CNN_Prot_Filters'],
                                        parameters['CNN_Prot_Filters_W'],'relu','same',parameters['Num_CNN_SMILES'],
                                        parameters['CNN_SMILES_Filters'],parameters['CNN_SMILES_Filters_W'],'relu','same',
                                        parameters['FCNN_Layers'],'relu',parameters['FCNN_Layers_Units'],parameters['Dropout'],
                                        'linear', tf.keras.optimizers.Adam(learning_rate=parameters['LR']),'mean_squared_error',
                                        [tf.keras.metrics.RootMeanSquaredError(),c_index]).fit_model(data_train,
                                        32,1000,callbacks,True,data_val)
                                                                                                     
        mse,rmse,ci=cnn_fcnn_model.evaluate([data_val[0],data_val[1]],data_val[2])
        metrics_results[0].append(mse)
        metrics_results[1].append(rmse)
        metrics_results[2].append(ci)
        
        result_values = list(np.hstack([parameters['Num_CNN_Prot'],parameters['CNN_Prot_Filters'],
                     parameters['CNN_Prot_Filters_W'],parameters['Num_CNN_SMILES'],
                     parameters['CNN_SMILES_Filters'],parameters['CNN_SMILES_Filters_W'],
                     parameters['FCNN_Layers'],parameters['FCNN_Layers_Units'],
                     parameters['Dropout'],mse,rmse,ci,num_run]))
        

        
        save_func(save_file,result_values)
        
    result_values = list(np.hstack([parameters['Num_CNN_Prot'],parameters['CNN_Prot_Filters'],
                     parameters['CNN_Prot_Filters_W'],parameters['Num_CNN_SMILES'],
                     parameters['CNN_SMILES_Filters'],parameters['CNN_SMILES_Filters_W'],
                     parameters['FCNN_Layers'],parameters['FCNN_Layers_Units'],
                     parameters['Dropout'],np.mean(metrics_results[0]),np.mean(metrics_results[1]),
                     np.mean(metrics_results[2]),'Mean']))
        
        
    save_func(save_file,result_values)
    

        
    
if __name__ == '__main__':
    # Data
    data_path={'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')}
    
    # Dataset
    protein_data, smiles_data, kd_values = dataset_builder(data_path).transform_dataset('Sequence','SMILES','Kd',1400,72)
    
    
    # Train and Test Clusters
    dataset,_,_,clusters = dataset_builder(data_path).get_data()
    
    
    validation = False
    train = False
    evaluate = True
    
    
    if validation:
        parameters = {'Num_CNN_Prot':3,'CNN_Prot_Filters':[64,64,128],'CNN_Prot_Filters_W':[4,4,5],
                      'Num_CNN_SMILES':3,'CNN_SMILES_Filters':[64,64,128],'CNN_SMILES_Filters_W':[4,4,5],
                      'FCNN_Layers':3,'FCNN_Layers_Units':[1024,512,1024],'Dropout':[0.5,0.1],
                      'LR':0.0001}
        
        
        grid_search(parameters,[protein_data,smiles_data,kd_values],
                    [list(clusters[i][1].iloc[:,0]) for i in range(len(clusters)) if clusters[i][0]!='test'],
                    '../gs_results/',
                    '../gs_results/grid_results.csv')
    
        
        
    if train:
        train_idx = pd.concat([i.iloc[:,0] for t,i in clusters if t=='train'])
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
    
        prot_train = tf.gather(protein_data, train_idx)
        prot_test = tf.gather(protein_data, test_idx)
    
        smiles_train = tf.gather(smiles_data, train_idx)
        smiles_test = tf.gather(smiles_data, test_idx)
    
        kd_train = tf.gather(kd_values, train_idx)
        kd_test = tf.gather(kd_values, test_idx)
        
        # Callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_root_mean_squared_error',
                                          min_delta = 0.001, patience = 50, mode = 'min', 
                                          restore_best_weights=True)
        
    
        mc = tf.keras.callbacks.ModelCheckpoint(filepath = '../model/cnn_model_pad_same', 
                                                monitor = 'val_root_mean_squared_error',
                                            save_best_only=True, save_weights_only = False, mode = 'min')
    
    
        callbacks = [es,mc]
        
        
        cnn_fcnn_model = CNN_FCNN_Model(1400,72,'one_hot',20,26,None,None,3,[64,64,128],[4,4,5],
                                'relu','same',3,[64,64,128],[4,4,5],'relu','same',3,'relu',[1024,512,1024],
                                [0.5,0.1],'linear', tf.keras.optimizers.Adam(learning_rate=0.0001),'mean_squared_error',
                                [tf.keras.metrics.RootMeanSquaredError(),c_index]).fit_model((prot_train,smiles_train,kd_train),
                                                                                              32,1000,callbacks,True,
                                                                                              (prot_test,smiles_test,kd_test))
        
                                                                                             

            
    if evaluate:
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
        prot_test = tf.gather(protein_data, test_idx)
        smiles_test = tf.gather(smiles_data, test_idx)
        kd_test = tf.gather(kd_values, test_idx)
        
        train_cnn_fcnn_model = tf.keras.models.load_model('../model/cnn_fcnn_model_pad_same',
                                            custom_objects={'c_index':c_index}) 
    
        metrics_results = inference_metrics(train_cnn_fcnn_model,[prot_test,smiles_test,kd_test])
        pred_scatter_plot(kd_test,train_cnn_fcnn_model.predict([prot_test,smiles_test]),
                          'Davis Kinase Dataset: Predictions vs True Values','True Values','Predictions',False,'')

        
        

        
        
        

