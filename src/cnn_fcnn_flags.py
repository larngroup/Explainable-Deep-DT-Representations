# -*- coding: utf-8 -*-

import os
import time
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
          '--prot_len',
          type=int,
          default = 1400,
          help = 'Protein Sequence Max Length'
          )
    
    parser.add_argument(
          '--smiles_len',
          type=int,
          default = 72,
          help ='SMILES String Max Length'
          )    
    
    
    parser.add_argument(
          '--prot_dict_len',
          type=int,
          default = 20,
          help = 'Protein AA Dictionary Size'
          )   

    parser.add_argument(
          '--smiles_dict_len',
          type=int,
          default = 26,
          help = 'SMILES Char Dictionary Size'
          )
    
    parser.add_argument(
        '--pad_opt',
        type = str,
        default = 'same',
        help = 'CNN Padding Option'
        )
 
    parser.add_argument(
        '--cnn_atv',
        type = str,
        default = 'relu',
        help = 'CNN Activation Function'
        )
    
    parser.add_argument(
        '--fcnn_atv',
        type = str,
        default = 'relu',
        help = 'MLP Dense Activation Function'
        )
    
    parser.add_argument(
        '--out_atv',
        type = str,
        default = 'linear',
        help = 'Output Dense Activation Function'
        )
    
    parser.add_argument(
        '--loss_fn',
        type = str,
        default = 'mean_squared_error',
        help = 'Model Loss Function'
        )  
    
    parser.add_argument(
      '--num_cnn_layers_prot',
      type=int,
      nargs='+',
      help='Number of Protein CNN Layers'
      )
    parser.add_argument(
      '--prot_filters',
      type=int,
      nargs='+',
      action='append',
      help='Number of filters for each Protein CNN Layer'
      )
    parser.add_argument(
      '--prot_filters_w',
      type=int,
      nargs='+',
      action='append',
      help='Filters Window for each Protein CNN Layers'
      )
    parser.add_argument(
      '--num_cnn_layers_smiles',
      type=int,
      nargs='+',
      help='Number of SMILES CNN Layers'
      )
    parser.add_argument(
      '--smiles_filters',
      type=int,
      nargs='+',
      action='append',
      help='Number of filters for each SMILES CNN Layer'
      )
    
    parser.add_argument(
      '--smiles_filters_w',
      type=int,
      nargs='+',
      action='append',
      help='Filters Window for each Protein CNN Layers'
      )
    
    parser.add_argument(
      '--num_fcnn_layers',
      type=int,
      nargs='+',
      help='Number of Dense Layers for the Prediction MLP'
      )
    
    parser.add_argument(
      '--fcnn_units',
      type=int,
      nargs='+',
      action = 'append',
      help='Output MLP Block Hidden Neurons'
      )
    
    parser.add_argument(
      '--drop_rate',
      type=float,
      nargs='+',
      action = 'append',
      help='Dropout Rate'
      )
    
    parser.add_argument(
      '--lr_rate',
      type=float,
      nargs='+',
      help='Optimizer Learning Rate'
      )
    parser.add_argument(
      '--num_epochs',
      type=int,
      default=500,
      help='Number of Epochs.'
      )
    parser.add_argument(
      '--batch_size',
      type=int,
      default=32,
      help='Batch Size'
      )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Log Directory'
        )

    parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Checkpoint Directory'
      )
    
    parser.add_argument(
      '--option',
      type=str,
      help = 'Train, Validation or Evaluation'
      )
    
    
    flags, unparsed = parser.parse_known_args()

    return flags


def logging(x, FLAGS):
    log_path = os.path.join( FLAGS.log_dir, "log.txt" )
    
    with open( log_path, "a" ) as log:
        log.write("%s\n" % x)
    print("------------------------//------------------------")
    print(x)
    print("------------------------//------------------------")

