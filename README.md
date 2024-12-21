# Stock price prediction

## Motivation

A stock market index gets affected by the news headlines that run daily on television and newspapers along with a multitude of other factors. Highly negative news impacts the stock market negatively and positive news impacts the stock market positively. This observation inspires to design and build a deep learning neural network which analyze news headlines and predict next-day stock prices for a specific market.
 
Here, we'll try to model this relationship between the news and the stock market price of an index. Our assumption in modelling the stock price in this exercise is that news headlines that run on a particular day affect the opening stock price of an index the very next morning.

## Table of Contents

- [System Overview](#system_overview)
- [Dataset Prepration](#general-information)
- [Model Architecture](#model-architecture)
- [Model Building](#model-architecture)
- [Model Summary](#model-summary)
- [Model Evaluation](#model-evaluation)
- [Technologies Used](#technologies-used)
- [Collaborators](#collaborators)

<!-- You can include any other section that is pertinent to your problem -->

## System Design

![basic](resources/basic_intent.png)


## Dataset Prepration

The primary challenge of building deep learning model is sourcing specific dataset for a given problem, and curating and preparing accurate dataset.

For this project, datasets from Kaggle, specifically Dow Jones stock prices and Reddit news headlines, have been utilized. The dataset contains historical data from 2008 to 2016.  The stock prices and the relevant news of each day has to be combined to form one dataset to feed into the network.

Lets look at a datapoint for a particular day from the datasets

News as of 2016-07-01 from the News dataset is given below:

![basic](resources/News_2016_07_01.png)


DowJones Stock market index as of 2016-07-01 from DowJones dataset is given below:


`Date,Open,High,Low,Close,Volume,Adj Close`

 `2016-07-01,17924.240234,18002.380859,17916.910156,17949.369141,82160000,17949.369141`

These two datasets needs to be combined by using date column to make one single dataset to be used for training the model. Stock market opening price is used as label while News of previous day is used as features to train the model. Once the model is trained, News of a particular day can be fed into the model to predict the stock market opening price. This is illustrated in the diagram below

![basic](resources/textmining.png)

## Model Architecture

### The CNN-RNN architecture

In this architecure, combination of Convolutional 1D and Recurrent Neural Network is used. 1D CNN to extract meaningful features which results in much shorter sequences in a much faster way. You can then feed this vector to an RNN in the same way as one would feed a sentence.

![cnn-rnn](resources/CNN_RNN_Architecure.jpg)

As you can see in the architecture, multiple convolutional layers are applied in parallel to the 'feature representation' of the text. The feature representation of the text is represented using glove vector representation. The output of the multiple convolutional layers are concatenated and RNN layer works on the top of it. 
 
A fully-connected layer sits on the top of RNN and since it is regression problem, there will be no activation to bound the output. 

The below image is the snapshot of the actual model architecture taken from Keras sequential model summary. We can tweak the model architecure with different combination of CNN convolutional, dropout, pooling layers and RNN layers to build better performing model

![cnn-rnn](resources/model_summary.png)


![cnn-rnn](resources/cnn-1d-rnn.jpg)


## Problem statement & Reference Architecture

- Aim: Use Reddit News Headlines to predict the movement of Dow Jones Industrial Average.

- Data Source: https://www.kaggle.com/aaron7sun/stocknews

- Data Description: Dow Jones details on Open, High, Low and Close for each day from 2008-08-08 to 2016-07-01 and headlines for those dates from Reddit News.

- Methodology: For this project, we will use GloVe to create our word embeddings and CNNs followed by LSTMs to build our model. This model is based off the work done in this paper https://www.aclweb.org/anthology/C/C16/C16-1229.pdf.



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
