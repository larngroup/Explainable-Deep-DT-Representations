# XDTA : Explainable Deep Drug-Target Representations for Binding Affinity Prediction
<p align="justify"> We explore the reliability of Convolutional Neural Networks (CNNs) in the identification of important regions for binding, and the significance of the deep representations by providing explanations to the model’s decisions based on the identification of the input regions that contributed the most to the prediction. Furthermore, we propose an end-to-end deep learning architecture to predict binding affinity, where CNNs are exploited in their capacity to automatically surmise and extract discriminating deep representations from 1D sequential and structural data.</p>

## End-to-End Deep Learning Architecture: Convolutional Neural Networks + Feed-Forward Fully Connected Neural Network
<p align="center"><img src="/figure/pred_model.png"/></p>

## Chemogenomc Representative K-Fold 
<p align="center"><img src="/figure/split_data.png"/></p>

## Regression Discriminative Localization Map
<p align="center"><img src="/figure/gradram_result.png"/></p>

## Binding Affinity Prediction Models
- **Two Parallel Convolution Neural Networks + Fully Connected Neural Network** (Deep Representations)
- **Random Forest Regressor** (Deep Representations)
- **Support Vector Regressor** (Deep Representations)
- **Gradient Boosting Regressor** (Deep Representations)
- **Kernel Ridge Regression** (Deep Representations)

## Gradient-Weighted Regression Activation Mapping 
- **Global Max Pooling + Guided Gradients**
- **Global Max Pooling + Non Guided Gradients**
- **Global Average Pooling + Guided Gradients**
- **Global Average Pooling + Non Guided Gradients**

## Dictionaries
- **davis_prot_dictionary**: AA char-integer dictionary
- **davis_smiles_dictionary**: SMILES char-integer dictionary
