# Explainable Deep Drug-Target Representations for Binding Affinity Prediction
<p align="justify"> We explore the reliability of Convolutional Neural Networks (CNNs) in the identification of important regions for binding, and the significance of the deep representations by providing explanations to the model’s decisions based on the identification of the input regions that contributed the most to the prediction. Furthermore, we implement an end-to-end deep learning architecture to predict binding affinity, where CNNs are exploited in their capacity to automatically surmise and extract discriminating deep representations from 1D sequential and structural data.</p>

## End-to-End Deep Learning Architecture: Convolutional Neural Networks + Feed-Forward Fully Connected Neural Network
<p align="center"><img src="/figure/pred_model.png"/></p>

## Chemogenomc Representative K-Fold 
<p align="center"><img src="/figure/split_data.png"/></p>

## Regression Discriminative Localization Map
<p align="center"><img src="/figure/gradram_result.png"/></p>

## 3D Docking Visualization
**ABL1(E255K)-phosphorylated - SKI-606**
<p align="center"><img src="/figure/abl1_ski606.png"/></p>
**DDR1 - Erlotinib**
<p align="center"><img src="/figure/ddr1_erlotinib.png"/></p>

## Binding Affinity Prediction Models
- **Two Parallel Convolution Neural Networks + Fully Connected Neural Network** (Deep Representations)
- **Random Forest Regressor** (Deep Representations)
- **Support Vector Regressor** (Deep Representations)
- **Gradient Boosting Regressor** (Deep Representations)
- **Kernel Ridge Regression** (Deep Representations)

## Gradient-Weighted Regression Activation Mapping (Grad-RAM)
- **Global Max Pooling + Guided Gradients**
- **Global Max Pooling + Non Guided Gradients**
- **Global Average Pooling + Guided Gradients**
- **Global Average Pooling + Non Guided Gradients**

## Davis Kinase Binding Affinity
### Dataset
- **davis_original_dataset:** original dataset
- **davis_dataset_processed:** dataset processed : prot sequences + rdkit SMILES strings + pkd values
- **deep_features_dataset:** CNN deep representations: protein + SMILES deep representations
### Clusters
- **test_cluster:** independent test set indices
- **train_cluster_X:** train indices 
### Similarity
- **protein_sw_score:** protein Smith-Waterman similarity scores
- **protein_sw_score_norm:** protein Smith-Waterman similarity normalized scores
- **smiles_ecfp6_tanimoto_sim:** SMILES Morgan radius 3 similarity scores
### Binding
- **davis_scpdb_binding:** davis-scpdb matching pairs binding information
### PSSM
- **pssm_X:** davis-scpdb matching pairs PSSM

## sc-PDB Pairs
### Binding
- **scpdb_binding:** scpdb pairs binding information
### PSSM
- **pssm_X:** scpdb pairs PSSM

## Dictionaries
- **davis_prot_dictionary**: AA char-integer dictionary
- **davis_smiles_dictionary**: SMILES char-integer dictionary

## Requirements:
### Python
- Python 3.7.9
- Tensorflow 2.4.1
- Numpy 
- Pandas
- Scikit-learn
- Itertools
- Matplotlib
- Seaborn
- Glob
- Json
- Pickle
- RDKIT (Env)
### R
- Biostrings
