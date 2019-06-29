'''
Created on February 4, 2017
Modified on June 14, 2019

@author: optas
@modify: Tod
'''

import tensorflow as tf
import numpy as np
import warnings

from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.conv import conv_1d, avg_pool_1d
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import fully_connected, dropout

from utils.tf import expand_scope_by_name, replicate_parameter_for_all_layers

def variational_encoder_with_convs_and_symmetry(in_signal, n_filters=[64, 128, 256, 1024], filter_sizes=[1], strides=[1],
                                        b_norm=True, non_linearity=tf.nn.relu, regularizer=None, weight_decay=0.001,
                                        symmetry=tf.reduce_max, dropout_prob=None, pool=avg_pool_1d, pool_sizes=None, scope=None,
                                        reuse=False, padding='same', verbose=False, closing=None, conv_op=conv_1d):
    '''An Encoder (recognition network), which maps inputs onto a latent space.
    '''

    if verbose:
        print('Building Encoder')

    n_layers = len(n_filters)
    filter_sizes = replicate_parameter_for_all_layers(filter_sizes, n_layers)
    strides = replicate_parameter_for_all_layers(strides, n_layers)
    dropout_prob = replicate_parameter_for_all_layers(dropout_prob, n_layers)

    if n_layers < 2:
        raise ValueError('More than 1 layers are expected.')
        
    def _one_layer(layer, i):
        name = 'encoder_conv_layer_' + str(i)
        scope_i = expand_scope_by_name(scope, name)
        layer = conv_op(layer, nb_filter=n_filters[i], filter_size=filter_sizes[i], strides=strides[i], regularizer=regularizer,
                        weight_decay=weight_decay, name=name, reuse=reuse, scope=scope_i, padding=padding)

        if verbose:
            print(name, 'conv params = ', np.prod(layer.W.get_shape().as_list()) + np.prod(layer.b.get_shape().as_list()), end=' ')

        if b_norm:
            name += '_bnorm'
            scope_i = expand_scope_by_name(scope, name)
            layer = batch_normalization(layer, name=name, reuse=reuse, scope=scope_i)
            if verbose:
                print('bnorm params = ', np.prod(layer.beta.get_shape().as_list()) + np.prod(layer.gamma.get_shape().as_list()))

        if non_linearity is not None:
            layer = non_linearity(layer)

        if pool is not None and pool_sizes is not None:
            if pool_sizes[i] is not None:
                layer = pool(layer, kernel_size=pool_sizes[i])

        if dropout_prob is not None and dropout_prob[i] > 0:
            layer = dropout(layer, 1.0 - dropout_prob[i])

        if verbose:
            print(layer)
            print('output size:', np.prod(layer.get_shape().as_list()[1:]), '\n')
            
        return layer
    
    for i in range(n_layers - 1):
        if i == 0:
            layer = in_signal
        layer = _one_layer(layer, i)
            
    i += 1
    z_mean = _one_layer(layer, i)
    z_std = _one_layer(layer, i)

    if symmetry is not None:
        z_mean = symmetry(z_mean, axis=1, name='z_mean')
        z_std = symmetry(z_std, axis=1, name='z_std')
        if verbose:
            print(z_mean)
            print(z_std)

    if closing is not None:
        z_mean = closing(z_mean, name='z_mean')
        z_std = closing(z_std, name='z_std')
        print(z_mean)
        print(z_std)

    return z_mean, z_std