import sys
import sys

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
from keras.models import Model
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from sklearn.cluster import KMeans

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_tensor=Input(shape=(224, 224, 3)))

print(base_model.summary())

net = Model(input=base_model.input, output=base_model.get_layer('global_average_pooling2d_1').output)

labeled_hdf5_path = 'datasets/veri/veri-labeled-data.hdf5'
unlabeled_hdf5_path = 'datasets/veri/veri-unlabeled-data.hdf5'

labelled_images = HDF5Matrix(labeled_hdf5_path, 'train_images')
labelled_labels = HDF5Matrix(labeled_hdf5_path, 'train_labels')

unlabelled_images = HDF5Matrix(unlabeled_hdf5_path, 'train_images')
unlabelled_labels = HDF5Matrix(unlabeled_hdf5_path, 'train_labels')

labelled_images = np.array(labelled_images)
labelled_labels = np.array(labelled_labels)

unlabelled_images = np.array(unlabelled_images)
unlabelled_labels = np.array(unlabelled_labels)

# session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# use GPU to calculate the similarity matrix
center_t = tf.placeholder(tf.float32, (None, None))
other_t = tf.placeholder(tf.float32, (None, None))
center_t_norm = tf.nn.l2_normalize(center_t, dim=1)
other_t_norm = tf.nn.l2_normalize(other_t, dim=1)
similarity = tf.matmul(center_t_norm, other_t_norm, transpose_a=False, transpose_b=True)

# extract features
features = []
for img in labelled_images:
    feature = net.predict(img)
    features.append(np.squeeze(feature))

features = np.array(features)

print("Features Shapes", features.shape)

# clustering
kmeans = KMeans(n_clusters=5).fit(features)

# select centers
distances = kmeans.transform(features)  # num images * NUM_CLUSTER
center_idx = np.argmin(distances, axis=0)
centers = [features[i] for i in center_idx]

# calculate similarity matrix
similarities = sess.run(similarity, {center_t: centers, other_t: features})  # NUM_CLUSTER * num images

# select reliable images
reliable_image_idx = np.unique(np.argwhere(similarities > .98)[:, 1])
print('ckpt %d: # reliable images %d' % (0, len(reliable_image_idx)))

sys.stdout.flush()

images = np.array([unlabeled_images[i][0] for i in reliable_image_idx])
labels = to_categorical([kmeans.labels_[i] for i in reliable_image_idx])
