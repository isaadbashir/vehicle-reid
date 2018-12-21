import numpy as np
from keras.utils.io_utils import HDF5Matrix
from models.resnet import ResNet
from trainer.train_gradually import TrainGradually

labeled_hdf5_path = 'datasets/veri/veri-labeled-data.hdf5'
unlabeled_hdf5_path = 'datasets/veri/veri-unlabeled-data.hdf5'

number_of_classes = 5
model = ResNet(number_of_classes, 'imagenet', False, (224, 224, 3))

labelled_images = HDF5Matrix(labeled_hdf5_path, 'train_images')
labelled_labels = HDF5Matrix(labeled_hdf5_path, 'train_labels')

unlabelled_images = HDF5Matrix(unlabeled_hdf5_path, 'train_images')
unlabelled_labels = HDF5Matrix(unlabeled_hdf5_path, 'train_labels')

labelled_images = np.array(labelled_images)
labelled_labels = np.array(labelled_labels)

unlabelled_images = np.array(unlabelled_images)
unlabelled_labels = np.array(unlabelled_labels)

trainer = TrainGradually(model.get_resnet_model(), (labelled_images, labelled_labels), (unlabelled_images, unlabelled_labels), 'knn')

# trainer.train_loop_loose(10, 0.001, 8, 1)
trainer.train_loop_strict(10, 0.001, 8, 1)
