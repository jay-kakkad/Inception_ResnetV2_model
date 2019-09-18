import tensorflow as tf
import tf_utils as util
import numpy as np

def stem(net,is_training=False,activation_fn=tf.nn.relu,scope = None,reuse = None,bn_decay=0.9):
    output = None
    with tf.variable_scope(scope,'Stem',[net],reuse = reuse):
        stem = util.conv2d(net,32,[3,3],scope='1_conv2d_1',stride = [2,2],padding = 'VALID',use_xavier = True,is_training = is_training,bn=True)
        stem = util.conv2d(stem,32,[3,3],scope='1_conv2d_2',stride = [1,1],padding = 'VALID',use_xavier = True,is_training = is_training,bn=True)
        stem = util.conv2d(stem,64,[3,3],scope='1_conv2d_3',stride = [1,1],padding = 'SAME',use_xavier = True,is_training = is_training,bn=True)

        with tf.variable_scope(scope,'Level-2',[stem],reuse=reuse):
            layer2_1 = util.conv2d(stem,96,[3,3],scope = '2_Conv2d',stride = [2,2] , padding = 'VALID',use_xavier = True,is_training = is_training,bn=True)
            layer2_2 = util.max_pool2d(stem,[3,3],scope = '2_MaxPool',stride = [2,2] , padding='VALID')
            stem = util.filter_concat(3,[layer2_1,layer2_2])

        with tf.variable_scope(scope,'Level-3',[stem],reuse=reuse):
            with tf.variable_scope('branch_1'):   
                layer3_1_1 = util.conv2d(stem,64,[1,1],scope = '3/1_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
                layer3_1_2 = util.conv2d(layer3_1_1,96,[3,3],scope = '3_conv_3x3',stride = [1,1],padding='VALID',use_xavier = True,bn=True,is_training=is_training)
            with tf.variable_scope('branch_2'):
                layer3_2_1 = util.conv2d(stem,64,[1,1],scope = '3/2-conv_1x1',stride = [1,1],padding = 'SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
                layer3_2_2 = util.conv2d(layer3_2_1,64,[7,1],scope = '3/2_conv_7x1',stride = [1,1],padding='SAME',use_xavier = True,bn=True,is_training=is_training)
                layer3_2_3 = util.conv2d(layer3_2_2,64,[1,7],scope = '3/2_conv_1x7',stride = [1,1],padding='SAME',use_xavier = True,bn=True,is_training=is_training)
                layer3_2_4 = util.conv2d(layer3_2_3,96,[3,3],scope = '3/2_conv_3x3',stride = [1,1],padding='VALID',use_xavier = True,bn=True,is_training=is_training)
            stem=util.filter_concat(3,[layer3_1_2,layer3_2_4])
        with tf.variable_scope('Level-4'):
            layer4_1 = util.conv2d(stem,192,[3,3],scope = '4_Conv2d',stride = [2,2] , padding = 'VALID',use_xavier = True,bn=True,is_training = is_training)
            layer4_2 = util.max_pool2d(stem,[3,3],scope = '4_MaxPool',stride = [2,2] , padding='VALID')
            stem = util.filter_concat(3,[layer4_1,layer4_2])
        output = stem
    return output

def block35(net,is_training=False,activation_fn=tf.nn.relu,scope = None,reuse = None,scale=.2):
    output = None
    with tf.variable_scope(scope,'Inception-A',[net],reuse = reuse):
        with tf.variable_scope('branch_1'):
            layer1_1 = util.conv2d(net,32,[1,1],scope = '5/1_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
        with tf.variable_scope('branch_2'):
            layer2_1 = util.conv2d(net,32,[1,1],scope = '5/2_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
            layer2_2 = util.conv2d(layer2_1,32,[3,3],scope = '5/2_conv_3x3',stride = [1,1],padding='SAME',use_xavier = True,activation_fn=activation_fn,bn=True)
        with tf.variable_scope('branch_3'):
            layer3_1 = util.conv2d(net,32,[1,1],scope = '5/3_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
            layer3_2 = util.conv2d(layer3_1,48,[3,3],scope = '5/3_conv_3x3_1',stride = [1,1],padding='SAME',use_xavier = True,bn=True,activation_fn=activation_fn)
            layer3_3 = util.conv2d(layer3_2,64,[3,3],scope = '5/3_conv_3x3_2',stride = [1,1],padding='SAME',use_xavier = True,bn=True,activation_fn=activation_fn)
            
        concat1 = util.filter_concat(3,[layer1_1,layer2_2,layer3_3])
        
        concat2 = util.conv2d(concat1,net.get_shape()[3].value,[1,1],scope = '5_concat',stride = [1,1],padding = 'SAME',use_xavier = True , activation_fn = None,bn=None,stddev = None)
        concat2 = concat2 * scale	
        shortcut = net + concat2
        
        shortcut = activation_fn(shortcut)
		
        output = shortcut
    return output
	
	
def block17(net,is_training=False,activation_fn=tf.nn.relu,scope = None,reuse = None,scale = 0.2):
    output = None
    with tf.variable_scope(scope,'Inception-B',[net],reuse = reuse):
        with tf.variable_scope('branch_1'):
            layer1_1 = util.conv2d(net,192,[1,1],scope = '7/1_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
        with tf.variable_scope('branch_2'):
            layer2_1 = util.conv2d(net,192,[1,1],scope = '7/2_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
            layer2_2 = util.conv2d(layer2_1,160,[1,7],scope = '7/2_conv_1x7',stride = [1,1],padding='SAME',use_xavier = True,activation_fn=activation_fn,bn = True)
            layer2_3 = util.conv2d(layer2_2,192,[7,1],scope = '7/2_conv_7x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn=activation_fn,bn = True)
        concat = util.filter_concat(3,[layer1_1,layer2_3])
        concat = util.conv2d(concat,net.get_shape()[3].value,[1,1],scope = '7_concat',stride = [1,1],padding = 'SAME',use_xavier = True , activation_fn = None,bn=None,stddev = None)
        concat = concat * scale
        shortcut = net + concat
        shortcut = activation_fn(shortcut)
		
        output = shortcut
       
    return output


def block8(net,is_training=False,activation_fn=tf.nn.relu,scope = None,reuse = None,scale = 0.2):
    output = None
    with tf.variable_scope(scope,'Inception-C',[net],reuse = reuse):
        with tf.variable_scope('branch_1'):
            layer1_1 = util.conv2d(net,192,[1,1],scope = '9/1_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
        with tf.variable_scope('branch_2'):
            layer2_1 = util.conv2d(net,192,[1,1],scope = '9/2_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True)
            layer2_2 = util.conv2d(layer2_1,160,[1,7],scope = '9/2_conv_1x7',stride = [1,1],padding='SAME',use_xavier = True,activation_fn=activation_fn,bn = True)
            layer2_3 = util.conv2d(layer2_2,192,[7,1],scope = '9/2_conv_7x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn=activation_fn,bn = True)
        concat = util.filter_concat(3,[layer1_1,layer2_3])
        concat = util.conv2d(concat,net.get_shape()[3].value,[1,1],scope = '9_concat',stride = [1,1],padding = 'SAME',use_xavier = True , activation_fn = None,bn=None,stddev = None)
        concat = concat * scale
        shortcut = net + concat
        shortcut = activation_fn(shortcut)
		
        output = shortcut
    return output
	

def reduce35(net,activation_fn=tf.nn.relu,scope = None,reuse = None,is_training=False):
	output = None
	with tf.variable_scope(scope,'Reduction-A',[net],reuse=reuse):
		with tf.variable_scope('branch_1'):
			layer1_1 = util.max_pool2d(net,[3,3],scope = '6/1_MaxPool',stride = [2,2] , padding='VALID')
   
		with tf.variable_scope('branch_2'):
			layer2_1 = util.conv2d(net,384,[3,3],scope = '6/2_conv_3x3',stride = [2,2],padding='VALID',use_xavier = True,bn=True,is_training=is_training)
		with tf.variable_scope('branch_3'):
			layer3_1 = util.conv2d(net,256,[1,1],scope = '6/3_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
			layer3_2 = util.conv2d(layer3_1,256,[3,3],scope = '6/3_conv_3x3_1',stride = [1,1],padding='SAME',use_xavier = True,bn=True,is_training=is_training)
			layer3_3 = util.conv2d(layer3_2,384,[3,3],scope = '6/3_conv_3x3_2',stride = [2,2],padding='VALID',use_xavier = True,bn=True,is_training=is_training)
		concat= util.filter_concat(3,[layer1_1,layer2_1,layer3_3])
		output = concat
	return output
 		

	
def reduce17(net,activation_fn=tf.nn.relu,scope = None,reuse=None,is_training=False):
    output = None
    with tf.variable_scope(scope,'Reduction-B',[net],reuse=reuse):
        with tf.variable_scope('branch_1'):
            layer1_1 = util.max_pool2d(net,[3,3],scope = '8/1_MaxPool',stride = [2,2] , padding='VALID')
        with tf.variable_scope('branch_2'):
            layer2_1 = util.conv2d(net,256,[1,1],scope = '8/2_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
            layer2_2 = util.conv2d(layer2_1,384,[3,3],scope = '8/2_conv_3x3',stride = [2,2],padding = 'VALID',use_xavier=True,bn=True,is_training=is_training)
        with tf.variable_scope('branch_3'):
            layer3_1 = util.conv2d(net,256,[1,1],scope = '8/3_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
            layer3_2 = util.conv2d(layer3_1,288,[3,3],scope = '8/3_conv_3x3',stride=[2,2],padding='VALID',use_xavier=True,bn=True,is_training=is_training)
        with tf.variable_scope('branch_4'):
            layer4_1 = util.conv2d(net,256,[1,1],scope = '8/4_conv_1x1',stride = [1,1],padding='SAME',use_xavier = True,activation_fn = None,bn=True,is_training=is_training)
            layer4_2 = util.conv2d(layer4_1,288,[3,3],scope = '8/4_conv_3x3_1',stride=[1,1],padding='SAME',use_xavier=True,bn=True,is_training=is_training)
            layer4_3 = util.conv2d(layer4_2,320,[3,3],scope = '8/4_conv_3x3_2',stride=[2,2],padding='VALID',use_xavier=True,bn=True,is_training=is_training)
        concat= util.filter_concat(3,[layer1_1,layer2_2,layer3_2,layer4_3])
		
        output = concat
	
    return output
"""
def fully_connected_layer(net,scope=None,
                    bn=True,is_training=False,reuse=None):
    output=None
    with tf.variable_scope(scope,'Fully_connected',[net],reuse=reuse):
        layer1 = util.fully_connected(net,128,'fc_1',use_xavier=True,bn=True,is_training=is_training)    
#        layer2 = util.fully_connected(layer1,144,'fc_2',use_xavier=True,bn=True,is_training=is_training)
        
    
#        layer3 = util.fully_connected(layer2,18,'fc_3',use_xavier=True,bn=True,is_training=is_training)
        
#        layer4 = util.fully_connected(layer3,2,'fc_4',use_xavier=True,bn=True,is_training=is_training,activation_fn=None)
        
        layer = tf.layers.dense(inputs=layer1, units=2)
        
        output=layer
    
    return output
    

def avg_pooling(inputs,reuse=None,is_training=True,bn=True,scope=None):
    output=None
    with tf.variable_scope(scope,'Avg_Pool',[inputs],reuse=reuse):
        avg_pool_layer = util.avg_pool2d(inputs,[8,8],'avg_pool_3x3',[1,1])
#        layer = util.conv2d(avg_pool_layer, 128, [1,1], scope='Conv2d_1b_1x1',padding='VALID',is_training=is_training,bn=bn)
#        layer = util.conv2d(layer, 768, [layer.get_shape()[1].value,layer.get_shape()[2].value],
#                                    padding='VALID', scope='Conv2d_2a_5x5',is_training=is_training,bn=bn)
        flatten = tf.contrib.layers.flatten(avg_pool_layer)
        dense =  tf.layers.dense(inputs=flatten, units=2048, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.8,training = is_training)
        output=dropout
    return output
"""
def model_fn(inputs,is_training=False,bn=False,reuse=None):
	with tf.variable_scope('Inception_Resnet_V2') as sc:
		net = stem(inputs,is_training=True,reuse=reuse,scope=sc)
		net = block35(net,is_training=True,reuse=reuse,scope=sc)
		net = reduce35(net,reuse=reuse,scope=sc)
		net = block17(net,is_training=True,reuse=reuse,scope=sc)
		net = reduce17(net,reuse=reuse,scope=sc)
		net = block8(net,is_training=True,reuse=reuse,scope=sc)
		return net
    		
      

	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	