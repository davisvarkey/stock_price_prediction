# Stock price prediction


## Problem statement & Reference Architecture

- Aim: Use Reddit News Headlines to predict the movement of Dow Jones Industrial Average.

- Data Source: https://www.kaggle.com/aaron7sun/stocknews

- Data Description: Dow Jones details on Open, High, Low and Close for each day from 2008-08-08 to 2016-07-01 and headlines for those dates from Reddit News.

- Methodology: For this project, we will use GloVe to create our word embeddings and CNNs followed by LSTMs to build our model. This model is based off the work done in this paper https://www.aclweb.org/anthology/C/C16/C16-1229.pdf.

![basic](resources/basic_intent.png)

## Table of Contents

- [General Info](#general-information)
- [Model Architecture](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.

![datasetgraph](./class_distribution.png)

In order to address the challenge of class imbalance, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.

## Pictorial representation of skin types

![skincancertypes](./skin_cancer_types.png)

The aim of this task is to assign a specific class label to a particular type of skin cancer.

## Model Architecture

### The CNN-RNN architecture

![cnn-rnn](resources/cnn-1d-rnn.jpg)

## Model Summary

![Model Architecture](./mode-1-summary.png)

## Model 1 Evaluation

![Model Evaluation](./model-1-evaluation.png)


### **Note:**  Model 4 trained with augmented data, without batch normalization and a dropout layer before flatterning, that resulted well-balanced performance, displaying no signs of underfitting or overfitting.

## Overall Observations:

- The implementation of class rebalancing has notably enhanced the model's performance across both training and validation datasets.
- The narrow divergence between training and validation accuracies underscores the robust generalization capability of the final CNN model.
- The addition of batch normalization failed to enhance both training and validation accuracy.
- Those classes have significant percentage of representation were predicted accurately and on the other hand those classes which have low representation were predicted incorrectly.
- 'basal cell carcinoma' class which has 376 representation (third highest) predicted correctly.
- 'melanoma' class which has 438 representation (second highest) predicted correctly.
- 'nevus' class which has 357 representation (fourth highest) predicted correctly.
- 'pigmented benign keratosis' class which has 462 representation (highest) predicted correctly.

## Technologies Used

- [Python](https://www.python.org/) - version 3.11.4
- [Matplotlib](https://matplotlib.org/) - version 3.7.1
- [Numpy](https://numpy.org/) - version 1.24.3
- [Pandas](https://pandas.pydata.org/) - version 1.5.3
- [Seaborn](https://seaborn.pydata.org/) - version 0.12.2
- [Tensorflow](https://www.tensorflow.org/) - version 2.15.0

## Collaborators

Created by [@davisvarkey](https://github.com/davisvarkey)
