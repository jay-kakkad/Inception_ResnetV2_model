# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:52:02 2018

@author: jpkak
"""

import os
import time
import tensorflow as tf
import modelv2 as mods
import glob
#-------------------DATASET-INFORRMATION--------------------------------------#
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

img_h,img_w = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
batch_size = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172

train_path='C:/Innefu/data/training_set'
test_path='C:/Innefu/data/test_set'

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(inputs=base_model.input, outputs=predictions)
  return model

train_datagen =  ImageDataGenerator(
      rescale = 1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
test_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_w,img_h),
    batch_size=batch_size,
  )

validation_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_w,img_h),
    batch_size=batch_size,
  )

print(str(train_generator))
x = tf.placeholder(tf.float32,[None,299,299,3])
reuse1=None
model = mods.model_fn(x,is_training=False,bn=True)
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)
sess.run(model,feed_dict={x:train_generator})




