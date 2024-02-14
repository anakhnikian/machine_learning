# python_keras_demo

## Functions: 

This is a small collection of programs I've used to training and test deep learning models to classify data collected by a collaborator. New users of 
keras and tensorflow can use this code as an example of integrating multiple features of those packages into a basic analysis pipeline. More 
experienced users can build on this backbone to produce more sophisticated analytic frameworks.

## Example Using Iris Data

The following demonstration code uses the iris dataset (https://archive.ics.uci.edu/dataset/53/iris), included in the repository

import pandas as pd
from obj.network_params import network_obj
from model_builder import build_model
from stats.train_validate import stratified_split

dataframe= pd.read_csv("iris.data", header=None)
dataset = dataframe.values

X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

NN = network_obj(n_features = 4,n_labels = 3,n_hidden = 2, units_hidden = [10,5])
NN_L2 = build_model(NN,reg_type = 'L2')



iris_skf = stratified_split(NN_L2, loss = 'categorical_crossentropy', features=X, target=Y)

## Dependencies

tensorflow

keras

scikit-learn

numpy

pandas
