# -*- coding: utf-8 -*-
"""

@author: NelsonRCM
"""
import time
from dataset_builder_util import *
from cnn_fcnn_model import *
import itertools
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.svm import SVR as svr
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.kernel_ridge import KernelRidge as krr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score  as r2s
import glob
import pickle
from scipy import stats


def extract_cnn_representations(protein_data, smiles_data, model,
                                prot_input_layer = 'Protein_Input', smiles_input_layer = 'SMILES_Input', 
                                prot_cnn_layer = 'Prot_Global_Max', smiles_cnn_layer= 'SMILES_Global_Max'):
    

    
    deep_representations = tf.keras.Model(inputs = [model.get_layer(prot_input_layer).input, 
                                                    model.get_layer(smiles_input_layer).input],
                                          outputs = [model.get_layer(prot_cnn_layer).output,
                                                     model.get_layer(smiles_cnn_layer).output],
                                          name = 'deep_representations')
    
    
    
    # prot_features = tf.keras.Model(model.get_layer(prot_input_layer).input, 
    #                                model.get_layer(prot_cnn_layer).output, 
    #                                name='prot_features')
    
    # smiles_features = tf.keras.Model(model.get_layer(smiles_input_layer).input, 
    #                                model.get_layer(smiles_cnn_layer).output, 
    #                                name='smiles_features')
    
    prot_features,smiles_features = deep_representations([protein_data,smiles_data])
    
    return prot_features,smiles_features


def custom_grid_search(model,data,data_folds):
    n_folds = len(data_folds)
    metrics_results = []
    for i in range(n_folds):
        print('Fold: ' + str(i))
        index_train = list(itertools.chain.from_iterable([data_folds[j] for j in range(len(data_folds)) if j!=i]))
        index_val = data_folds[i]

        data_train = data[0].iloc[index_train,:]
        target_train = data[1].iloc[index_train]
        
        data_val = data[0].iloc[index_val,:]
        target_val = data[1].iloc[index_val]

        model.fit(data_train,target_train)
        pred_values = model.predict(data_val)
        
        metrics_results.append([mse(target_val,pred_values),mse(target_val,pred_values,squared=False),
                        c_index(target_val,pred_values).numpy(), r2s(target_val,pred_values),
                        stats.spearmanr(target_val,pred_values)[0]])


    metrics_results = np.mean(np.array(metrics_results),axis=0)

    return metrics_results 


def inference_metrics(model,data):
    start = time.time()
    pred_values = model.predict(data[0])
    end = time.time()
    inf_time = end-start

    metrics = {'MSE': mse(data[1],pred_values), 'RMSE': mse(data[1],pred_values,squared=False),
               'CI':c_index(data[1],pred_values).numpy(), 'R2': r2s(data[1],pred_values),
               'Spearman':stats.spearmanr(data[1],pred_values)[0],'Time':inf_time}
    
    return metrics
    
def model_cv(model_type,parameters,data,data_folds):
    parameters_combination = list(itertools.product(*parameters))
    results = {}
    
    print('Model Type: ' + model_type)
    if model_type == 'RFR':
        for i in range(len(parameters_combination)):
            print('Parameter Combination: '+ str(i) + str(' out of ')+ str(len(parameters_combination)))
        
            rfr_model = rfr(n_estimators=parameters_combination[i][0], criterion='mse',
                            max_features = 'auto', n_jobs = -1)
            
            metric_results = custom_grid_search(rfr_model,data,data_folds)
        
            results[i] = {'Model':'RFR','N Estimators':parameters_combination[i][0],
                      'MSE': metric_results[0], 'RMSE' : metric_results[1], 'CI': metric_results[2], 'R2': metric_results[3],
                      'Spearman':metric_results[4]}
        
        
        
    elif model_type == 'SVR':
        for i in range(len(parameters_combination)):
            print('Parameter Combination: '+ str(i) + str(' out of ')+ str(len(parameters_combination)))            
            svr_model = svr(kernel=parameters_combination[i][0], degree = parameters_combination[i][1],
                            gamma = 'scale', C = parameters_combination[i][2]) 
                            

            metric_results = custom_grid_search(svr_model,data,data_folds)
        
            results[i] = {'Model':'SVR','kernel':parameters_combination[i][0],'degree': parameters_combination[i][1],
                      'C':  parameters_combination[i][2],'MSE': metric_results[0], 'RMSE' : metric_results[1],
                      'CI': metric_results[2], 'R2': metric_results[3],'Spearman':metric_results[4]}        
        
        
        
    elif model_type == 'GBR':
        for i in range(len(parameters_combination)):
            print('Parameter Combination: '+ str(i) + str(' out of ')+ str(len(parameters_combination)))        
            gbr_model = gbr(loss=parameters_combination[i][0], learning_rate = parameters_combination[i][1],
                            n_estimators = parameters_combination[i][2], criterion = parameters_combination[i][3],
                            max_features = 'auto')
            
            metric_results = custom_grid_search(gbr_model,data,data_folds)
        
            results[i] = {'Model':'GBR','loss': parameters_combination[i][0], 'LR':parameters_combination[i][1],
                          'N Estimators':parameters_combination[i][2],'Criterion':parameters_combination[i][3]
                          ,'MSE': metric_results[0], 'RMSE' : metric_results[1],
                          'CI': metric_results[2], 'R2': metric_results[3],'Spearman':metric_results[4]}        
            
            
    elif model_type == 'KRR':
        for i in range(len(parameters_combination)):
            print('Parameter Combination: '+ str(i) + str(' out of ')+ str(len(parameters_combination)))        
            krr_model = krr(alpha=parameters_combination[i][0], kernel = parameters_combination[i][1], 
                              degree = parameters_combination[i][2])
            metric_results = custom_grid_search(krr_model,data,data_folds)
        
            results[i] = {'Model':'KRR','Alpha':parameters_combination[i][0],'Kernel': parameters_combination[i][1],
                          'Degree': parameters_combination[i][2],
                          'MSE': metric_results[0], 'RMSE' : metric_results[1],'CI': metric_results[2],
                          'R2': metric_results[3],'Spearman':metric_results[4]} 
        

        
    return results
    
               
         
if __name__ == '__main__':
    
    # Davis Dataset
    
    data_path={'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')}
    
    cnn_model = tf.keras.models.load_model('../model/cnn_fcnn_model_pad_same',
                                            custom_objects={'c_index':c_index}) 
    
    
    protein_data, smiles_data, kd_values = dataset_builder(data_path).transform_dataset('Sequence','SMILES','Kd',1400,72)
    
    # protein_rep = []
    # smiles_rep = []
    
    # for i in range(len(protein_data)):
    #     print(i)
    #     p_r,s_r = extract_cnn_representations(tf.gather(protein_data,i)[None,:],tf.gather(smiles_data,i)[None,:],cnn_model)
    #     protein_rep.append(p_r)
    #     smiles_rep.append(s_r)
    
    # # protein_rep,smiles_rep = extract_cnn_representations(protein_data,smiles_data,cnn_model)
    
    # deep_representations = tf.squeeze(tf.stack([tf.concat([protein_rep[i],smiles_rep[i]],axis=1)
    #                                             for i in range(len(protein_rep))]),axis=1)
    
    
    
    deep_representations = pd.read_csv('../data/davis/dataset/deep_features_dataset.csv',header=None)
    
    
    
    _,_,_,clusters = dataset_builder(data_path).get_data()
    
    data_clusters=[list(i[1].iloc[:,0]) for i in clusters if i[0]!='test']
    

    
    validation = False
    train = False
    evaluate = True
    
    if validation:
        rfr_gs_cv = model_cv('RFR',[[i for i in range(100,1000,200)]], [pd.DataFrame(deep_representations),
                                                                    pd.Series(kd_values)],data_clusters)
    
        svr_gs_cv = model_cv('SVR',[['linear','poly','rbf','sigmoid'],[3,4,5],[0.001,0.01,0.1,1,3,5]],
                              [deep_features,pd.Series(kd_values)],data_clusters)
    
        gbr_gs_cv = model_cv('GBR',[['ls','lad'],[0.001,0.01,0.1],[i for i in range(100,1000,200)],['friedman_mse', 'mse']],
                                    [pd.DataFrame(deep_representations),pd.Series(kd_values)],data_clusters)
    
        rls_gs_cv = model_cv('RLS',[[0.01,0.1,0.3,0.5,0.7,1,3,5],['svd','cholesky','sparse_cg','lsqr','sag']],
                              [pd.DataFrame(deep_representations),pd.Series(kd_values)],data_clusters)
    
        krr_gs_cv = model_cv('KRR',[[0.01, 0.1,0.3,0.5,0.7,1,3,5],['rbf'],
                                      [3]], [pd.DataFrame(deep_representations),pd.Series(kd_values)],data_clusters)
    
    
    if train:
        train_idx = pd.concat([i.iloc[:,0] for t,i in clusters if t=='train'])
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
    
        # # RFR
        rfr_model = rfr(n_estimators=300, criterion='mse', max_features = 'auto', n_jobs = -1)
        rfr_model.fit(pd.DataFrame(deep_representations).iloc[train_idx,:],pd.Series(kd_values).iloc[train_idx])
        rfr_model_results = inference_metrics(rfr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        # pickle.dump(rfr_model,open('../model/rfr_model.py','wb'))
    
        # # SVR
        svr_model = svr(kernel='rbf', gamma = 'scale', C = 5) 
        svr_model.fit(pd.DataFrame(deep_representations).iloc[train_idx,:],pd.Series(kd_values).iloc[train_idx])
        svr_model_results = inference_metrics(svr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        # pickle.dump(svr_model,open('../model/svr_model.py','wb'))
    
        # # GBR
        gbr_model = gbr(loss='ls', learning_rate = 0.1,
                    n_estimators = 900, criterion = 'friedman_mse',
                    max_features = 'auto')
        gbr_model.fit(pd.DataFrame(deep_representations).iloc[train_idx,:],pd.Series(kd_values).iloc[train_idx])
        gbr_model_results = inference_metrics(gbr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        
        # pickle.dump(gbr_model,open('../model/gbr_model.py','wb'))
    
    
        # # KRR
        krr_model = krls(alpha=0.01, kernel = 'poly', degree = 5)
        krr_model.fit(pd.DataFrame(deep_representations).iloc[train_idx,:],pd.Series(kd_values).iloc[train_idx])
        krr_model_results = inference_metrics(krr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                            pd.Series(kd_values).iloc[test_idx]])
        # pickle.dump(krr_model,open('../model/krr_model.py','wb'))
        
        
        
    if evaluate:
        test_idx = [i for t,i in clusters if t=='test'][0].iloc[:,0]
        rfr_model = pickle.load(open('../model/rfr_model.py','rb'))
        rfr_model_results = inference_metrics(rfr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        
        svr_model = pickle.load(open('../model/svr_model.py','rb'))
        svr_model_results = inference_metrics(svr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        
        gbr_model = pickle.load(open('../model/gbr_model.py','rb'))
        gbr_model_results = inference_metrics(gbr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                          pd.Series(kd_values).iloc[test_idx]])
        
        krr_model = pickle.load(open('../model/krr_model.py','rb'))
        krr_model_results = inference_metrics(krr_model, [pd.DataFrame(deep_representations).iloc[test_idx,:],
                                                            pd.Series(kd_values).iloc[test_idx]])

    