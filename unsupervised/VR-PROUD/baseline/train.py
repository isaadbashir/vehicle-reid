from __future__ import division, print_function, absolute_import

import tensorflow as tf
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.io_utils import HDF5Matrix

hdf5_path = '../dataset/VeRi/hdf5/veri-dataset.hdf5'  # address to where you want to save the hdf5 file

images = HDF5Matrix(hdf5_path, 'train_img')
labels = HDF5Matrix(hdf5_path, 'train_labels')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
set_session(sess)

# load pre-trained resnet50
base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

x = base_model.output
x = Flatten(name='flatten')(x)
x = Dropout(0.5)(x)
x = Dense(770, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
net = Model(input=base_model.input, output=x)

for layer in net.layers:
   layer.trainable = True

# train
batch_size = 16
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=20, # 0. 
    width_shift_range=0.2, # 0.
    height_shift_range=0.2,# 0.
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    data_format=K.image_data_format())

net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')
net.fit_generator(datagen.flow(images, labels, batch_size=batch_size), steps_per_epoch=len(images)/batch_size+1, epochs=40)
net.save('base.ckpt')
