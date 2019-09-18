# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:26:51 2017

@author: jpkak
"""

import tensorflow as tf
import numpy as np
#---Defining-Variables---------------------------------------------------------------------------------------------------------------#	
#------------------------------------------------------------------------------------------------------------------------------------#	
def _variable_on_cpu(name,shape,intializer,use_fp16=False):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name,shape,initializer=intializer,dtype=dtype)
    return var
#------------------------------------------------------------------------------------------------------------------------------------#	
def _variable_with_weight_decay(name,shape,stddev,wd,use_xavier=True):
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev = stddev)
    var = _variable_on_cpu(name,shape,initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)
    return var
#----Defining-Layers-----------------------------------------------------------------------------------------------------------------#	
#------------------------------------------------------------------------------------------------------------------------------------#	
def conv1d(inputs,num_output_channels,kernel_size,scope,stride=1,padding='SAME',use_xavier = True
           ,stddev = 1e-3,weight_decay = 0.0,activation_fn=tf.nn.relu,bn=False
           ,bn_decay=None,
           is_training=None):
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,num_in_channels,num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape = kernel_shape,
                                             use_xavier= use_xavier,
                                             stddev= stddev,
                                             wd = weight_decay)
        outputs = tf.nn.conv1d(inputs,kernel,stride=stride,padding = padding)
        biases = _variable_on_cpu('biases',[num_output_channels],tf.constant_initializer(0.0))
        
        outputs = tf.nn.bias_add(outputs,biases)
        
        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs   
#------------------------------------------------------------------------------------------------------------------------------------#	
def conv2d(inputs,num_output_channels,kernel_size,scope,stride=[1,1],padding='SAME',use_xavier = True
           ,stddev = 1e-3,weight_decay = 0.0,activation_fn=tf.nn.relu,bn=False
           ,bn_decay=None,
           is_training=False):
    with tf.variable_scope(scope) as sc:
        kernel_h,kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h,kernel_w,num_in_channels,num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
        stride_h,stride_w = stride
        outputs = tf.nn.conv2d(inputs,kernel,[1,stride_h,stride_w,1],
                                   padding = padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
            
        outputs = tf.nn.bias_add(outputs,biases)
        
        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')
        
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs       
#def conv2d_transpose():-------------------------------------------------------------------------------------------------------------#	    
def conv3d(inputs,num_output_channels,kernel_size,scope,stride=[1,1,1],padding='SAME',use_xavier = True
           ,stddev = 1e-3,weight_decay = 0.0,activation_fn=tf.nn.relu,bn=False
           ,bn_decay=None,
           is_training=None):
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)
        
        if bn:
          outputs = batch_norm_for_conv3d(outputs, is_training,
                                          bn_decay=bn_decay, scope='bn')
    
        if activation_fn is not None:
          outputs = activation_fn(outputs)
        return outputs	
#------------------------------------------------------------------------------------------------------------------------------------#	
def fully_connected(inputs,num_outputs,scope,use_xavier=True,stddev = 1e-3,weight_decay = 0.0,
                    activation_fn=tf.nn.relu,bn=False,bn_decay=None,
                    is_training = None):
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs
#------------------------------------------------------------------------------------------------------------------------------------#	
def max_pool2d(inputs,kernel_size,scope,stride = [2,2],padding = 'VALID'):
    with tf.variable_scope(scope) as sc:
        kernel_h,kernel_w = kernel_size
        stride_h,stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize = [1,kernel_h,kernel_w,1],
                                 strides = [1,stride_h,stride_w,1],
                                 padding = padding,name = sc.name)
        return outputs
#------------------------------------------------------------------------------------------------------------------------------------#	   
def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs
#------------------------------------------------------------------------------------------------------------------------------------#		
def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs	
#------------------------------------------------------------------------------------------------------------------------------------#	
def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs
#------------------------------------------------------------------------------------------------------------------------------------#		
#------------------------------------------------------------------------------------------------------------------------------------#	
def batch_norm_template(inputs,is_training,scope,moments_dims,bn_decay):
    with tf.variable_scope(scope):
        is_training = tf.cast(is_training,tf.bool)
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())
        
        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
          with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
        
        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed
#------------------------------------------------------------------------------------------------------------------------------------#	
def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
	return batch_norm_template(inputs, is_training, scope, [0], bn_decay)
#------------------------------------------------------------------------------------------------------------------------------------#	
#------------------------------------------------------------------------------------------------------------------------------------#	
def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
	return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay)
#------------------------------------------------------------------------------------------------------------------------------------#
def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
	return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay)
#------------------------------------------------------------------------------------------------------------------------------------#	
def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
	return batch_norm_template(inputs, is_training, scope, moment_dims=[0,1,2,3],bn_decay=bn_decay)
#------------------------------------------------------------------------------------------------------------------------------------#	
#------------------------------------------------------------------------------------------------------------------------------------#		
def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
	with tf.variable_scope(scope) as sc:
		outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
		return outputs	
#------------------------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------------------------#		
def filter_concat(axis,values):
	return tf.concat(values,axis)
 
        