This project is targeted for survival prediction on cardivascular data. 

## classification folder
This folder treats a survival problem as different classification problems. Unclear labels, 
i.e., censoring before targeting time is deleted. The purpose is to compare the performance 
difference of neural network and logistic regression on the dataset.

classification/logistic.py : logistic regression
classification/nn.py: neural network to perform classification

## dsn folder
This folder is implementation of https://peerj.com/articles/6257/ 
In this framework, survival times are discretized into several buckets. And loglikelihood 
is used as loss function. 

dsn/dsn2.py: discretize time into two buckets. This is same as classifcation problem but 
take censoring into account
dsn/dsn5.py: discretize time into five buckets. This aligns with original implementation 
in the paper
dsn/dsnfull.py: discretize time by year. This is the most useful survival model since it 
predicts survival probability every year.

## main folder
common.py: all definition of feature lists (MARKERS, BASE_COVS etc)
data.py: generation of data from original lrpp_updated.csv
loss.py: all loss functions and evaluation metric as c-index and aucJM
models.py: all definition of neural networks
preprocess.py: summarize features of original dataset
r_connection.py: compare aucJM and c-index function from exported R results
utils.py: some utility functions 

## cox folder
This is a depreciated folder which is an (unsuccessful) implementation of 
https://arxiv.org/pdf/1705.10245.pdf