from __future__ import division, print_function, absolute_import

import os
import sys
import numpy as np
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from PIL import Image

DATASET_TEST = '/home/saad/dataset/VeRi/image_test'
DATASET_QUERY = '/home/saad/dataset/VeRi/image_query'

TEST = '../dataset/veri/name_test_partial.txt'
TEST_NUM = 5949
QUERY = '../dataset/veri/name_query_partial.txt'
QUERY_NUM = 893


def extract_feature(dir_path, net, folder):
  features = []
  infos = []
  num = 0
  with open(dir_path, 'r') as f:

    for image_name in f:
      arr = image_name.strip()
      arr = arr.split('_')
      carId = int(arr[0])
      camId = int(arr[1][1:])
      image_path = os.path.join(folder, image_name.strip())
      img = image.load_img(image_path, target_size=(224, 224))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      feature = net.predict(x)
      features.append(np.squeeze(feature))
      infos.append((carId, camId))

  return features, infos

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


# load model
net = load_model('base.ckpt')
net.summary()

net = Model(input=net.input, output=net.get_layer('avg_pool').output)

test_f, test_info = extract_feature(TEST, net, DATASET_TEST)
query_f, query_info = extract_feature(QUERY, net, DATASET_QUERY)