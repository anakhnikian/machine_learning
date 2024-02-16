#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:17:43 2023
For all routines NN is the network object returned by network_params and model_builder
@author: alexander
"""

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


def naive_fit(NN, loss, features, labels,
              optimizer = 'adam', metrics = ['accuracy'], 
              batch_size = 32, epochs = 100, verbose = 0,
              shuffle = False, validation_split = 0.3):
    model = NN.model
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_labels = encoder.transform(labels)
    onehot_labels = to_categorical(encoded_labels)
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    training = model.fit(features, onehot_labels, batch_size = batch_size, epochs=epochs, shuffle = shuffle, verbose = verbose, validation_split = validation_split)
    return training
    


def stratified_split(NN, loss, features, target, 
                     optimizer='adam', metrics=['accuracy'], 
                     batch_size=32, n_splits = 5, epochs=100, verbose=0):
    
    model = NN.model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target)
    onehot_target = to_categorical(encoded_target)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    lst_accu_stratified = []
    
    for train_index, test_index in skf.split(features, encoded_target):
        x_train_fold, x_test_fold = features[train_index], features[test_index]
        y_train_fold, y_test_fold = onehot_target[train_index], onehot_target[test_index]
        
        model.fit(x_train_fold, y_train_fold, batch_size=batch_size, epochs=epochs, verbose=verbose)
        score = model.evaluate(x_test_fold, y_test_fold)
        lst_accu_stratified.append(score[1])
        
    return lst_accu_stratified


def kfold_split(NN, features, target, loss, epochs = 200, batch_size = 32, n_splits = 5, verbose = 0, optimizer = 'adam', metrics = ['accurcy']):
    model = NN.model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    encoder = LabelEncoder()
    encoded_target = encoder.fit_transform(target)
    onehot_target = to_categorical(encoded_target)
    
    accuracies = []
    losses = []
    for train_index, test_index in kf.split(features):
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = onehot_target[train_index], onehot_target[test_index]
    
        # Fit the model on the training data
        model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)
    
        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(x_test, y_test)
    
        accuracies.append(accuracy)
        losses.append(loss)
        
    return accuracies


    
