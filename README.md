# Explainable Deep Drug-Target Representations for Binding Affinity Prediction
<p align="justify"> We explore the reliability of Convolutional Neural Networks (CNNs) in the identification of important regions for binding, and the significance of the deep representations by providing explanations to the model’s decisions based on the identification of the input regions that contributed the most to the prediction. Furthermore, we implement an end-to-end deep learning architecture to predict binding affinity, where CNNs are exploited in their capacity to automatically surmise and extract discriminating deep representations from 1D sequential and structural data.</p>

## End-to-End Deep Learning Architecture: Convolutional Neural Networks + Feed-Forward Fully Connected Neural Network
<p align="center"><img src="/figure/pred_model.png" width="70%" height="70%"/></p>

## Chemogenomc Representative K-Fold 
<p align="center"><img src="/figure/split_data.png" width="70%" height="70%"/></p>

## Regression Discriminative Localization Map
<p align="center"><img src="/figure/gradram_result.png" width="90%" height="90%"/></p>

## 3D Docking Visualization
- **Potential Binding Sites (≤ 5 Å) : Green**

- **L-Grad-RAM Hits : Blue**

- **Matched Binding - L-Grad-RAM Hits : Red**

### **ABL1(E255K)-phosphorylated - SKI-606**
<p align="center"><img src="/figure/abl1_ski606.png" width="90%" height="90%"/></p>

### **DDR1 - Foretinib**
<p align="center"><img src="/figure/ddr1_foretinib.png" width="90%" height="90%"/></p>

## Binding Affinity Prediction Model
- **Two Parallel Convolution Neural Networks + Fully Connected Neural Network**

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

## State-of-the-Art Baselines Data
Davis Kinase Binding Affinity Dataset + Clusters in the SOTA method format

## Docking
- **abl1_pymol.pse**: ABL1(E255K)-phosphorylated - SKI-606 PyMol Session
- **ddr1_pymol.pse**: DDR1 - Foretinib PyMol Session

## Requirements:
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

## Usage:
## Binding Affinity Prediction
### Training
```
python cnn_fcnn_model.py --option Training --num_cnn_layers_prot 3 --prot_filters 64 64 128 --prot_filters_w 4 4 5 --num_cnn_layers_smiles 3 --smiles_filters 64 64 128 --smiles_filters_w 4 4 5 --num_fcnn_layers 3 --fcnn_units 1024 512 1024 --drop_rate 0.5 0.1 --lr_rate 0.0001 
```
### Validation
```
python cnn_fcnn_model.py --option Validation --num_cnn_layers_prot 3 --prot_filters 64 64 128 --prot_filters_w 4 4 5 --num_cnn_layers_smiles 3 --smiles_filters 64 64 128 --smiles_filters_w 4 4 5 --num_fcnn_layers 3 --fcnn_units 1024 512 1024 --drop_rate 0.5 0.1 --lr_rate 0.0001 
```

### Evaluation
```
python cnn_fcnn_model.py --option Evaluation
```

##  Gradient-weighted Regression Activation Mapping (L-Grad-RAM)
**Example**
- **Protein Sequence** : MLEICLKLVG...
- **SMILES String** : Cc1cn(...
- **Window Length** : 0 1 2 ...
- **Feature Importance Threshold** : 0.3 0.4 0.5 ...
- **Binding Sites Positions** : 5 10 50 ...

```
python gradram_testing.py --protein_sequence MLEICLKLVG... --smiles_string Cc1cn(... --window 0 1 2 ... --thresholds 0.3 0.4 0.5 ... --sites 5 10 50 ...
```
