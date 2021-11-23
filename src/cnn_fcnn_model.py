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
import gc
from cnn_fcnn_flags import *
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score  as r2s
import time

def inference_metrics(model,data):
    start = time.time()
    pred_values = model.predict([data[0],data[1]])
    end = time.time()
    inf_time = end-start

    metrics = {'MSE': mse(data[2],pred_values), 'RMSE': mse(data[2],pred_values,squared=False),
               'CI':c_index(data[2],pred_values).numpy(), 'R2': r2s(data[2],pred_values),
               'Spearman':stats.spearmanr(data[2],pred_values)[0],'Time':inf_time}
    
    return metrics

def generate_one_hot_layer(input_dim, input_length,name=None):
    def one_hot_enconding(x, num_classes):
        one_hot = tf.one_hot(tf.cast(x, 'uint8'), depth=num_classes)
        one_hot = tf.gather(
            one_hot, [i for i in range(1, num_classes)], axis=2)
        return one_hot

    one_hot_layer = tf.keras.layers.Lambda(one_hot_enconding, arguments={
                                           'num_classes': input_dim}, input_shape=(input_length,),name=name)
    return one_hot_layer


def c_index(y_true,y_pred):
    matrix_pred=tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    matrix_pred=tf.cast(matrix_pred == 0.0, tf.float32) * 0.5 + tf.cast(matrix_pred > 0.0, tf.float32)
    
    
    matrix_true=tf.subtract(tf.expand_dims(y_true, -1), y_true)
    matrix_true=tf.cast(matrix_true>0.0,tf.float32)
    
    matrix_true_position=tf.where(tf.equal(matrix_true,1))
    
    matrix_pred_values=tf.gather_nd(matrix_pred,matrix_true_position)
    
    result=tf.where(tf.equal(tf.reduce_sum(matrix_pred_values),0),0.0,tf.reduce_sum(matrix_pred_values)/tf.reduce_sum(matrix_true))
    
    return result


def cnn_fcnn_model(FLAGS, cnn_layers_prot, prot_filters, prot_filter_w,
                   cnn_layers_smiles, smiles_filters,  
                   smiles_filter_w, num_fcnn_layers,
                   fcnn_units, dropout_rate, lr, metrics_list):
    
    prot_input = tf.keras.Input(shape=(FLAGS.prot_len,),dtype=tf.int64,name='Protein_Input')
    smiles_input = tf.keras.Input(shape=(FLAGS.smiles_len,),dtype=tf.int64, name='SMILES_Input')
    
    prot_emb = generate_one_hot_layer(FLAGS.prot_dict_len + 1, FLAGS.prot_len, name = 'Prot_Emb')(prot_input)
    smiles_emb = generate_one_hot_layer(FLAGS.smiles_dict_len + 1, FLAGS.prot_len, name = 'SMILES_Emb')(smiles_input)
    
    for i in range(cnn_layers_prot):
        prot_emb = tf.keras.layers.Conv1D(filters = prot_filters[i],
                                          kernel_size = prot_filter_w[i],
                                          strides = 1,
                                          padding = FLAGS.pad_opt,
                                          activation = FLAGS.cnn_atv,
                                          name = 'Prot_CNN_%d'%i)(prot_emb)
        
    for i in range(cnn_layers_smiles):
        smiles_emb = tf.keras.layers.Conv1D(filters = smiles_filters[i],
                                            kernel_size = smiles_filter_w[i],
                                            strides = 1,
                                            padding = FLAGS.pad_opt,
                                            activation = FLAGS.cnn_atv,
                                            name = 'SMILES_CNN_%d'%i)(smiles_emb)

    prot_out = tf.keras.layers.GlobalMaxPool1D(name ='Prot_Global_Max')(prot_emb)
    smiles_out = tf.keras.layers.GlobalMaxPool1D(name ='SMILES_Global_Max')(smiles_emb)


    deep_representations = tf.keras.layers.Concatenate(name='Concatenate')([prot_out,smiles_out])
    
    for i in range(num_fcnn_layers - 1):
        deep_representations = tf.keras.layers.Dropout(rate=dropout_rate[i], name='Dropout_%d'%i)(
                                tf.keras.layers.Dense(units=fcnn_units[i],activation=FLAGS.fcnn_atv,
                                                      name='Dense_%d'%i)(deep_representations))
        
    deep_representations = tf.keras.layers.Dense(units = fcnn_units[-1],activation = FLAGS.fcnn_atv,
                                                 name = 'Dense_%d'%(num_fcnn_layers-1))(deep_representations)
    
    out = tf.keras.layers.Dense(units = 1, activation = FLAGS.out_atv,
                                kernel_initializer='normal')(deep_representations)
    
    model = tf.keras.Model(inputs=[prot_input,smiles_input],outputs=out, name='CNN_FCNN_Model')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                  loss=FLAGS.loss_fn, metrics=metrics_list)
    
    return model
                                                 
def chemogenomic_folds_grid_search(FLAGS,data,folds,metrics_list):
    prot_cnn_layers_set = FLAGS.num_cnn_layers_prot
    prot_filters_set = FLAGS.prot_filters
    prot_filters_w_set = FLAGS.prot_filters_w
    
    
    smiles_cnn_layers_set = FLAGS.num_cnn_layers_smiles
    smiles_filters_set = FLAGS.smiles_filters
    smiles_filters_w_set = FLAGS.smiles_filters_w
    
    fcnn_layers_set = FLAGS.num_fcnn_layers
    fcnn_units_set = FLAGS.fcnn_units
    drop_rate_set = FLAGS.drop_rate
    
    lr_set = FLAGS.lr_rate
    

    logging("--------------------Grid Search-------------------",FLAGS)

    
    for params in itertools.product(prot_cnn_layers_set, prot_filters_set, prot_filters_w_set,
                                    smiles_cnn_layers_set, smiles_filters_set, smiles_filters_w_set,
                                    fcnn_layers_set, fcnn_units_set, drop_rate_set,lr_set):
        
        p1,p2,p3,p4,p5,p6,p7,p8,p9,p10 = params

        results = []

        
        for fold_idx in range(len(folds)):
            index_train=list(itertools.chain.from_iterable([folds[i] for i in range(len(folds)) if i!=fold_idx]))

            index_val=folds[fold_idx]
            
            data_train = [tf.gather(i,index_train) for i in data]
            
            data_val = [tf.gather(i,index_val) for i in data]
            
            es = tf.keras.callbacks.EarlyStopping(monitor = 'val_root_mean_squared_error',
                                         min_delta = 0.001, patience = 30, mode = 'min', 
                                         restore_best_weights=True)
            
            mc = tf.keras.callbacks.ModelCheckpoint(filepath = FLAGS.checkpoint_path+'/'+str(fold_idx)+'/',
                                                    monitor = 'val_root_mean_squared_error',
                                            save_best_only=True, save_weights_only = False, mode = 'min')

            
            grid_model = cnn_fcnn_model(FLAGS,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,metrics_list)

            grid_model.fit(x=[data_train[0],data_train[1]], y=data_train[2],
                                         batch_size = FLAGS.batch_size, epochs = FLAGS.num_epochs,
                                         verbose=2, callbacks = [es,mc],
                                         validation_data = ([data_val[0],data_val[1]],data_val[2]))
            
            preds = grid_model.predict([data_val[0],data_val[1]])
            metric_mse = mse(data_val[2],preds)
            metric_rmse = mse(data_val[2],preds,squared=False)
            metric_ci = c_index(data_val[2],preds).numpy()
            metric_r2 = r2s(data_val[2],preds)
            metric_spear = stats.spearmanr(data_val[2],preds)[0]
            
            results.append((metric_mse,metric_rmse,metric_ci,metric_r2,metric_spear))
            
            logging(("Prot CNN Layers = %d,  Prot Filters = %s, Prot Filters W = %s,"+
                     "SMILES CNN Layers = %d, SMILES Filters = %s, SMILES Filters W = %s, "+
                    "FCNN Layers = %d, FCNN Units = %s, Dropout Rate = %s, LR = %0.6f, " + 
                    "Fold = %d, MSE = %0.3f, RMSE = %0.3f, CI = %0.3f, R2 = %0.3f, Spear = %0.3f") % 
                    (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, fold_idx, metric_mse, metric_rmse, metric_ci, 
                     metric_r2, metric_spear),FLAGS)
            
        
        logging("Mean Folds - " + ((" MSE = %0.3f, RMSE = %0.3f, CI = %0.3f, R2 = %0.3f, Spear = %0.3f") %
                                   (np.mean(results,axis=0)[0], np.mean(results,axis=0)[1],
                                    np.mean(results,axis=0)[2],np.mean(results,axis=0)[3],
                                    np.mean(results,axis=0)[4])),FLAGS)
        
        
def run_train_val(FLAGS, data_path, metrics_list = [tf.keras.metrics.RootMeanSquaredError(),c_index]):
    
    protein_data, smiles_data, kd_values = dataset_builder(data_path).transform_dataset('Sequence','SMILES',
                                                                                        'Kd',FLAGS.prot_len,
                                                                                         FLAGS.smiles_len)
    
    _,_,_,clusters = dataset_builder(data_path).get_data()
    
    
    if FLAGS.option == 'Training':
        train_idx = pd.concat([i.iloc[:,0] for t,i in clusters if t=='train'])
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
    
        prot_train = tf.gather(protein_data, train_idx)
        prot_test = tf.gather(protein_data, test_idx)
    
        smiles_train = tf.gather(smiles_data, train_idx)
        smiles_test = tf.gather(smiles_data, test_idx)
    
        kd_train = tf.gather(kd_values, train_idx)
        kd_test = tf.gather(kd_values, test_idx)
        
        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_root_mean_squared_error',
                                          min_delta = 0.001, patience = 30, mode = 'min', 
                                          restore_best_weights=True)
        
    
        mc = tf.keras.callbacks.ModelCheckpoint(filepath = '../model/cnn_fcnn_model', 
                                                monitor = 'val_root_mean_squared_error',
                                            save_best_only=True, save_weights_only = False, mode = 'min')
        
        
        
        model = cnn_fcnn_model(FLAGS,FLAGS.num_cnn_layers_prot[0],FLAGS.prot_filters[0],
                       FLAGS.prot_filters_w[0],FLAGS.num_cnn_layers_smiles[0],
                       FLAGS.smiles_filters[0], FLAGS.smiles_filters_w[0],
                       FLAGS.num_fcnn_layers[0], FLAGS.fcnn_units[0], FLAGS.drop_rate[0],
                       FLAGS.lr_rate[0], metrics_list)
        
        model.fit(x=[prot_train,smiles_train], y=kd_train,
                  batch_size = FLAGS.batch_size, epochs = FLAGS.num_epochs,
                  verbose=2, callbacks = [es,mc],
                  validation_data = ([prot_test,smiles_test],kd_test))
        
        preds = model.predict([prot_test,smiles_test])
        metric_mse = mse(kd_test,preds)
        metric_rmse = mse(kd_test,preds,squared=False)
        metric_ci = c_index(kd_test,preds).numpy()
        metric_r2 = r2s(kd_test,preds)
        metric_spear = stats.spearmanr(kd_test,preds)[0]
        
        logging('Training Results:' + ("MSE = %0.3f, RMSE = %0.3f, CI = %0.3f, R2 = %0.3f, Spear = %0.3f") % 
                    (metric_mse, metric_rmse, metric_ci, metric_r2, metric_spear),FLAGS)
        
        
    if FLAGS.option == 'Validation':
        
        clusters = [list(clusters[i][1].iloc[:,0]) for i in range(len(clusters)) if clusters[i][0]!='test']
    
        chemogenomic_folds_grid_search(FLAGS,[protein_data,smiles_data,kd_values],clusters,metrics_list) 


                                               
    
 
if __name__ == '__main__':
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M",time.gmtime()) + "/"
    FLAGS.checkpoint_path = os.getcwd() + '/checkpoints/' + time.strftime("%d_%m_%y_%H_%M",time.gmtime()) + "/"
    
    data_path={'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')}

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
        
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

        
    
    if FLAGS.option == 'Training':
        logging(str(FLAGS), FLAGS)  
        run_train_val(FLAGS, data_path)
        
        
    elif FLAGS.option == 'Validation':
        logging(str(FLAGS), FLAGS)  
        run_train_val(FLAGS, data_path)
        
        
    elif FLAGS.option == 'Evaluation':
        protein_data, smiles_data, kd_values = dataset_builder(data_path).transform_dataset('Sequence','SMILES',
                                                                                        'Kd',FLAGS.prot_len,
                                                                                         FLAGS.smiles_len)
        
        train_cnn_fcnn_model = tf.keras.models.load_model('../model/cnn_fcnn_model_pad_same',
                                            custom_objects={'c_index':c_index})
        
        _,_,_,clusters = dataset_builder(data_path).get_data()
        
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
        prot_test = tf.gather(protein_data, test_idx)
        smiles_test = tf.gather(smiles_data, test_idx)
        kd_test = tf.gather(kd_values, test_idx)
        
        metrics_results = inference_metrics(train_cnn_fcnn_model,[prot_test,smiles_test,kd_test])
        print(metrics_results)
        pred_scatter_plot(kd_test,train_cnn_fcnn_model.predict([prot_test,smiles_test]),
                          'Davis Kinase Dataset: Predictions vs True Values','True Values','Predictions',False,'')
        
        

        
        

        
        
        

