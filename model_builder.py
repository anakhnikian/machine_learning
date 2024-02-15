# -*- coding: utf-8 -*-
"""
This module contains all required subroutines for building an ANN for single or multiclass classification
"""

from tensorflow.keras import models, layers, Input, Sequential, regularizers
from keras.layers import Dense
from utils.parse_parameters import get_act_functions, get_reg_params, get_unit_counts

#DEV NOTE: The indexing scheme for regularizing is needlessly complicated. 
#Just set drop rate or norm multiplier to zero for these


"""---------------------Mainfunction--------------------"""
def build_model(network_obj, reg_type = 'L2', reg_layers = 'All', reg_constant = 1e-4):
        
    """FUNCTION: Builds a simple neural network with L1 or L2 regularization and
    adds it to the class
    
    INPUTS:
        network_obj is the object returned by network_params
        
        reg_type is the regularization method (L1, L2, or drop, default: L2_
        
            L1 and L2 both amplify the relative influence of more salient 
            features by biasing weight vectors towards the direction of steeper
            gradient descent. L1 tends to completely surpress some features 
            (weights go to zero) while L2 reduces certain weights but in general
            does not produce vanishing solutions. Drop out randomly removes nodes 
            from each training iteration to simluate sampling from an ensemble
            of networks, thus reducing overfitting by making the network more
            flexible (less susceptible to fitting noise idiosyncratic to the
            training data)
            
        reg_layers is a list of layers to which the chosen regularization is 
        applied. default: 'All'
        
        reg_constant is the multiplier constant for the cost function offset
        in the case of L1 and L2, and the drop rate for drop out regularization.
        default value is 1e-4 and assumes regularization is L1 of L2. Must be in 
        [0,1] for drop out regularization       
        
        OUTPUT: 
            
        network_obj_new is the input object with new fields model and constant_values
        """
    
    #Update class attributes
    network_obj.reg_layers = reg_layers
    network_obj.reg_constant  = reg_constant
    network_obj.reg_type = reg_type
    

    #Extract parameters from input class
    n_features      = network_obj.n_features
    n_hidden        = network_obj.n_hidden
    units_hidden    = network_obj.units_hidden
    act_hidden      = network_obj.act_hidden
    n_labels        = network_obj.n_labels
    act_out         = network_obj.act_out
    
    #Convert class attrbutes to keras inputs as needed
    activations = get_act_functions(n_hidden, act_hidden)
    layer_units = get_unit_counts(units_hidden, n_hidden, n_features, n_labels)
    reg_indices, constant_values = get_reg_params(reg_layers, reg_constant, n_hidden)
    
    #DEV NOTE: act_hidden is redudant, remove from all code
    if reg_type == 'L1':
        model = get_L1_model(n_features, n_hidden, reg_indices, n_labels, constant_values, 
                         act_hidden, act_out, activations, layer_units)
    elif reg_type == 'L2': 
        model = get_L2_model(n_features, n_hidden, reg_indices, n_labels, constant_values, 
                         act_hidden, act_out, activations, layer_units)
    elif reg_type == 'drop':
        model = get_drop_model(n_features, n_hidden, reg_indices, n_labels, constant_values, 
                         act_out, activations, layer_units)
    else:
        raise Exception('Supported regularization is L1, L2, or drop')
            
       

    #add the model and return the modified class
    network_obj_new                 = network_obj
    network_obj_new.model           = model
    network_obj_new.reg_constants   = constant_values
    
    return network_obj_new

"""---------------------Mainfunction--------------------"""

"""------------------Subroutines for regularization types------------------"""
def get_L1_model(n_features, n_hidden, reg_indices, n_labels, constant_values, 
                 act_hidden, act_out, activations, layer_units):
    #Build layers
    inputs = Input(name = 'input', shape = (n_features,))

    h = [] #iniitialize hidden layer list
    for layer_idx in range(n_hidden):
        name        = 'h'+str(layer_idx+1)
        units       = layer_units[layer_idx]
        activation  = activations[layer_idx]
        
        #add layers
        if layer_idx == 0:
            h_current = Dense(name = name, units = units, 
                                     kernel_regularizer=regularizers.L1(constant_values[layer_idx]),
                                     activation = activation)(inputs)
            h.append(h_current)
        else:
            h_prior = h[layer_idx-1]
            h_current = Dense(name = name, units = units, 
                                     kernel_regularizer=regularizers.L1(constant_values[layer_idx]),
                                     activation = activation)(h_prior)
            h.append(h_current)

    
    #define output layer and put model together
    outputs = Dense(name = 'output', units = n_labels, activation = act_out)(h[-1])
    model  = models.Model(inputs=inputs, outputs=outputs)
    return model

def get_L2_model(n_features, n_hidden, reg_indices, n_labels, constant_values, 
                 act_hidden, act_out, activations, layer_units):
    #Build layers
    inputs = Input(name = 'input', shape = (n_features,))

    h = [] #iniitialize hidden layer list
    for layer_idx in range(n_hidden):
        name        = 'h'+str(layer_idx+1)
        units       = layer_units[layer_idx]
        activation  = activations[layer_idx]
        
        #add layers
        if layer_idx == 0:
            h_current = Dense(name = name, units = units, 
                                     kernel_regularizer=regularizers.L2(constant_values[layer_idx]),
                                     activation = activation)(inputs)
            h.append(h_current)
        else:
            h_prior = h[layer_idx-1]
            h_current = Dense(name = name, units = units, 
                                     kernel_regularizer=regularizers.L2(constant_values[layer_idx]),
                                     activation = activation)(h_prior)
            h.append(h_current)

    
    #define output layer and put model together
    outputs = Dense(name = 'output', units = n_labels, activation = act_out)(h[-1])
    model  = models.Model(inputs=inputs, outputs=outputs)
    return model

def get_drop_model(n_features=4, n_hidden = 2, reg_indices = [0,1], n_labels=3, constant_values = [0.2,0.2], 
                 act_out = 'softmax', activations = ['relu', 'relu'], layer_units = [10,5]):
    
    #Build layers
    inputs = Input(name = 'input', shape = (n_features,))
    rate_idx = 0 #initialize rate index, updates only when a drop layer is created
    h = [] #iniitialize hidden layer list
    for layer_idx in range(n_hidden):
        name        = 'h'+str(layer_idx+1)
        units       = layer_units[layer_idx]
        activation  = activations[layer_idx]
        
        #add layers
        if layer_idx == 0:
            h_current = Dense(name = name, units = units, activation = activation)(inputs)
            h.append(h_current)
        else:
            h_prior = h[layer_idx-1]
            h_current = Dense(name = name, units = units, activation = activation)(h_prior)
            h.append(h_current)
        #add dropout if defined
        if layer_idx in reg_indices:
            name = 'drop' + str(layer_idx+1)
            rate = constant_values[rate_idx]
            h[layer_idx] = layers.Dropout(name = name, rate = rate)(h[layer_idx])
            rate_idx += 1 #update rate index
    
    #define output layer and put model together
    outputs = Dense(name = 'output', units = n_labels, activation = act_out)(h[-1])
    model  = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

"""------------------Subroutines for regularization types------------------"""

def build_model_drop(network_obj, drop_layers = 'All', drop_rates = [0.2]):
    
    """FUNCTION: Builds a simple neural network with dropout regularization and
    adds it to the class
    
    INPUTS:
        """
    
    #Update class attributes
    network_obj.drop_layers = drop_layers
    network_obj.drop_rates  = drop_rates
    

    #Extract parameters from input class
    n_features      = network_obj.n_features
    n_hidden        = network_obj.n_hidden
    units_hidden    = network_obj.units_hidden
    act_hidden      = network_obj.act_hidden
    n_labels        = network_obj.n_labels
    act_out         = network_obj.act_out
    
    #Convert class attrbutes to keras inputs as needed
    activations = get_act_functions(n_hidden, act_hidden)
    layer_units = get_unit_counts(units_hidden, n_hidden, n_features, n_labels)
    reg_indices, constant_values = get_reg_params(drop_layers, drop_rates, n_hidden)
    
    
    #Build layers
    inputs = Input(name = 'input', shape = (n_features,))
    rate_idx = 0 #initialize rate index, updates only when a drop layer is created
    h = [] #iniitialize hidden layer list
    for layer_idx in range(n_hidden):
        name        = 'h'+str(layer_idx+1)
        units       = layer_units[layer_idx]
        activation  = activations[layer_idx]
        
        #add layers
        if layer_idx == 0:
            h_current = Dense(name = name, units = units, activation = activation)(inputs)
            h.append(h_current)
        else:
            h_prior = h[layer_idx-1]
            h_current = Dense(name = name, units = units, activation = activation)(h_prior)
            h.append(h_current)
        #add dropout if defined
        if layer_idx in reg_indices:
            name = 'drop' + str(layer_idx+1)
            rate = constant_values[rate_idx]
            h[layer_idx] = layers.Dropout(name = name, rate = rate)(h[layer_idx])
            rate_idx += 1 #update rate index
    
    #define output layer and put model together
    outputs = Dense(name = 'output', units = n_labels, activation = act_out)(h[-1])
    model  = models.Model(inputs=inputs, outputs=outputs)

    #add the model and return the modified class
    network_obj_new             =    network_obj
    network_obj_new.model           = model
    network_obj_new.reg_constants   = constant_values
    
    return network_obj_new

def baseline_model(n_features = 20, n_labels = 5, drop_rate = 0.2):
    model = Sequential(name="DeepNN", layers=[
    ### hidden layer 1
    Dense(name="h1", input_dim=n_features,
                 units=int(15), 
                 activation='relu'),
    layers.Dropout(name="drop1", rate= drop_rate),
    
    ### hidden layer 2
    Dense(name="h2", units=int(10), 
                 activation='relu'),
    layers.Dropout(name="drop2", rate= drop_rate),
    
    ### layer output
    Dense(name="output", units=n_labels, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model