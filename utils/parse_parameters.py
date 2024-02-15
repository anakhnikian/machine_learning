#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:38:19 2023

@author: alexander
"""
import numpy as np


## -------------CHECK PARAMETERS AND MODIFY AS NEEDED---------------------
def get_act_functions(n_hidden, act_hidden):
    """Converts default hidden layer activation settings to lists over which 
    to iteratively build sequential networks. Input class is unmodified if user 
    defined parameters are ready to pass to layers"""
    
    
    check_act_params(act_hidden, n_hidden) #check act_hidden match layer number
    
    #build activation list if needed
    if len(act_hidden) == 1: #constant activation function
        act_list = np.repeat(act_hidden, n_hidden)
    else:
        act_list = act_hidden #extract list from class attr
        
    return act_list
    

def get_reg_params(reg_layers, reg_constants, n_hidden):
    """Converts user inputs for regularization settings to lists for building 
    model"""
    
    #add parameter check here
    
    #build dropout list if needed
    if reg_layers == 'All': #constant across all layers
        reg_indices = np.arange(n_hidden+1) 
        constant_values = np.repeat(reg_constants,n_hidden)
    else: #if inputs are already lists
        reg_indices = reg_layers
        constant_values = reg_constants
    
    return reg_indices, constant_values

def get_unit_counts(units_hidden, n_hidden, n_features, n_labels):

    check_hidden_units(units_hidden, n_hidden, n_features, n_labels) #check unit counts match layers
    
    if units_hidden == 'Auto':
        unit_counts = []
        for layer_idx in range(n_hidden):
            layer_count = round(int((n_labels+n_features)/(2^layer_idx)))
            unit_counts.append(layer_count)
    else:
        unit_counts = units_hidden
            
    return unit_counts
                
                

## -------------RAISE EXCEPTIONS FOR INCOMPATIBLE INPUTS---------------------
def check_act_params(act_hidden, n_hidden):
    
    if 1 < len(act_hidden) < n_hidden: 
        raise Exception('act_hidden attribute must be a single str or list matching the number of hidden layers')
        
def check_drop_params(drop_layers, drop_rates, n_hidden):
    
    if 1 < len(drop_layers) < len(drop_rates):
        raise Exception('Number of drop layers must match number of rates')
    elif drop_layers == 'All' and 1 < len(drop_rates) < n_hidden:
        raise Exception('If all layers have drop outs drop rates must have length 1 or n_hidden')
        
def check_hidden_units(n_units, n_hidden, n_features, n_labels):
    
    if n_units == 'Auto' and 2^(n_hidden-1) >= (n_labels+n_features):
        raise Exception('Too many hidden layers for automatic unit allocation')
    elif n_units != 'Auto' and len(n_units) != n_hidden:
        raise Exception('If manually setting units there must be a number for each layer')