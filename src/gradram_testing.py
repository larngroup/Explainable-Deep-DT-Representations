# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 19:14:39 2021

@author: NelsonRCM
"""


import tensorflow as tf
from gradram import *
from gradram_eval_util import *
from dataset_builder_util import *
from cnn_fcnn_model import *
import itertools
import glob
from plots_util import *
import os
import sys
import time
import argparse
from PIL import Image


def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--protein_sequence',
        type=str,
        help='Protein Sequence'
        )
    
    parser.add_argument(
        '--smiles_string',
        type=str,
        help='SMILES String'
        )
    
    parser.add_argument(
        '--data_path',
        type=dict,
        default = {'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')},
        help='Data Path'
        )
    
    parser.add_argument(
        '--protein_layer',
        type=str,
        default = 'Prot_CNN_2',
        help='Protein Layer'
        )
    
    parser.add_argument(
        '--smiles_layer',
        type=str,
        default = 'Prot_CNN_2',
        help='SMILES Layer'
        )
    
    parser.add_argument(
        '--window',
        type=int,
        nargs='+',
        help='Window Values'
        )
    
    parser.add_argument(
        '--sites',
        type=int,
        nargs='+',
        help='Binding sites'
        )
    
    parser.add_argument(
        '--thresholds',
        type=float,
        nargs='+',
        help='Threshold Values'
        )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Log Directory'
        )
    
    flags, unparsed = parser.parse_known_args()

    return flags


def logging(msg, FLAGS):
    fpath = os.path.join( FLAGS.log_dir, "log.txt" )
    
    with open( fpath, "a" ) as fw:
        fw.write("%s\n" % msg)
    print("------------------------//------------------------")
    print(msg)
    print("------------------------//------------------------")

def gradram_eval_plot(FLAGS, model):
    
    _, prot_dict, smiles_dict, _ = dataset_builder(FLAGS.data_path).get_data()
    
    prot_data = tf.convert_to_tensor(dataset_builder(FLAGS.data_path).data_conversion(pd.Series(FLAGS.protein_sequence),
                                                                                      prot_dict,1400), dtype=tf.int64)
    
    smiles_data = tf.convert_to_tensor(dataset_builder(FLAGS.data_path).data_conversion(pd.Series(FLAGS.smiles_string),
                                                                                        smiles_dict,72), dtype=tf.int64)
    
    grad_eval_model = grad_ram(cnn_model, FLAGS.protein_layer, FLAGS.smiles_layer)

    gmp_gap = ['gmp', 'gap']
    guided = [True, False]
    
    ram_match = calculate_ram(grad_eval_model,prot_data,smiles_data,'gmp',True)
    
    
    if FLAGS.sites != None and FLAGS.window != None:

        match_results = [i[-1] for i in [matching_eval(matching_spot_window(ram_match[0],[FLAGS.sites],i)) 
                                         for i in FLAGS.window]]
    
        for i in range(len(FLAGS.window)):
            logging(('Window Size: '+str(FLAGS.window[i]) +' '+'Matching(%): '+str(match_results[i])),FLAGS)
    
    if FLAGS.sites != None and FLAGS.window != None and FLAGS.thresholds != None:
        feature_relevance = [[feature_rel_eval(matching_spot_window(ram_match[0],[FLAGS.sites],i),ram_match[0].numpy(),j)
                          for i in FLAGS.window] for j in FLAGS.thresholds]
    
    
        for i in range(len(FLAGS.thresholds)):
            for j in range(len(FLAGS.window)):
                logging(('Threshold: '+str(FLAGS.thresholds[i]*100)+' '+'Window Size: '+
                                    str(FLAGS.window[j]) +' '+'Matching(%): '+
                                    str(feature_relevance[i][j][-1])),FLAGS)

    
    
    gradram_plot([('Sequence',FLAGS.protein_sequence)],[ram_match[0][0,:]],[FLAGS.sites],
                  ['Prot'],['red'],['Binding Sites'],
                  ['o'],'Protein Sequence Length',True,FLAGS.log_dir+'prot_ram.png')


        
        
    im = Image.open(FLAGS.log_dir+'prot_ram.png') 
    im.show()
    
    
    
    



if __name__ == '__main__':

    data_path={'data':'../data/davis/dataset/davis_dataset_processed.csv',
            'prot_dic':'../dictionary/davis_prot_dictionary.txt',
            'smiles_dic':'../dictionary/davis_smiles_dictionary.txt',
            'clusters':glob.glob('../data/davis/clusters/*')}
    
    
    cnn_model = tf.keras.models.load_model('../model/cnn_fcnn_model_pad_same',
                                            custom_objects={'c_index':c_index})  
    
    cnn_model.trainable = False
    
    # CNN FCNN Model Last Conv Layers
    
    protein_layer = 'Prot_CNN_2'
    smiles_layer = 'SMILES_CNN_2'
    
    FLAGS = argparser()
    FLAGS.log_dir = os.getcwd() + '/logs/' + time.strftime("%d_%m_%y_%H_%M",time.gmtime()) + "/"
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    logging(str(FLAGS), FLAGS)
    
    
    gradram_eval_plot(FLAGS,cnn_model)
    



