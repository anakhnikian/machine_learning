#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FUNCTION: Initializes object to hold the network parameters and data to be passed
for building and training a simple neural network. 

All attributes except n_features and n_labels have default values

CLASS ATTRIBUTES:
    n_features: Length of feature vectors, assumed to be constant. 
    
    n_labels: Number of categories, for binary classification this equals 1 
    (single label wtih 'yes' or 'no' assignment for each vector)
    
    n_hidden: Number of hidden layers
        Values: scalar
        
        Default = 1
        
        Internal check will make sure the number of neurons in the final 
        hidden layer exceeds the number of labels. Set to 0 for a simple
        perceptron
        
    act_in: Activation function for input layer
    
        Value: Any valid method from TensorFlow libraries, or user defined 
        function
        
        Default = 'relu'
        
    act_hidden: Activation function(s) for hidden layers
    
        Values: List of activation functions for hidden layers. A single entry
        will set all layers to the same activation function
        
        Default = ['relu']
        
    act_out: Activation function for output layer
    
        Values: Same as act_in
        
        Default = 'sigmoid'
        
        
@author: A. Nakhnikian
"""

class network_obj:
    def __init__(self, n_features, n_labels, n_hidden = 1, units_hidden = 'Auto',
                 act_in = 'relu', act_hidden = ['relu'], act_out = 'Auto'):
        
        self.n_features     = n_features
        self.n_labels       = n_labels
        self.n_hidden       = n_hidden
        self.units_hidden   = units_hidden
        self.act_in         = act_in
        self.act_hidden     = act_hidden
        self.act_out        = act_out
        
        #Set defaults for output activation based on classifier type
        if act_out=='Auto' and n_labels>1:
            self.act_out = 'softmax'
        elif act_out=='Auto' and n_labels==1:
            self.act_out = 'sigmoid'
        #or set user-specified activation
        else:
            self.act_out = act_out
    
    @property
    def is_drop(self):
        """Indicator for drop regularization"""
        return True if self.reg_type=='drop' else False
    
    @property
    def is_L1(self):
        """Indicator for L1 regularization"""
        return True if self.reg_type=='L1' else False
    
    @property
    def is_L2(self):
        """Indicator for L2 regularization"""
        return True if self.reg_type=='L2' else False
    
    @property
    def has_model(self):
        """Indicator for model present"""
        return True if hasattr(self,'model') else False
        
        
    def add_data(self,frame):
        """Adds an attribute with a data frame"""
        self.data = frame
        self.has_data = True
        