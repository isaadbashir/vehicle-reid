import random

import h5py
import numpy as np
from datasets.dataset import Dataset
from utils.utils import Utils


class Veri(Dataset):
    def __init__(self, root_dir, train_folder, test_folder, query_folder, train_file, test_file, query_file, input_shape):
        super(Veri, self).__init__(root_dir, train_folder, test_folder, query_folder, train_file, test_file, query_file, input_shape)

    def read_train_data(self, image_loader, image_preprocessor, label_encoder):
        train_file_path = Utils.join_paths(self.root_dir, self.train_file)
        counter = 0
        total_images = Utils.count_item_in_file(train_file_path)
        images = []
        labels = []
        self.log.info("Starting to read the images from {} with total count of {}".format(train_file_path, total_images))

        change_label = -1

        with open(train_file_path, 'r', encoding='utf-8') as paths:
            for path in paths:
                path = Utils.pre_process_raw_string(path)
                lbl = Utils.get_label(self.log, path, '_', 0)

                if lbl != change_label:
                    self.train_identities = self.train_identities + 1
                    change_label = lbl

                img_path = Utils.join_paths(Utils.join_paths(self.root_dir, self.get_train_dir), path)
                img = image_loader.load_img(img_path, target_size=self.input_shape)
                img = image_loader.img_to_array(img)
                img = image_preprocessor(img)

                images.append(img)
                labels.append(self.train_identities - 1)

                Utils.printProgressBar(counter, total_images, prefix='Progress:', suffix=' ', length=50)
                counter = counter + 1

        self.original_labels = np.array(labels)
        labels = label_encoder(labels)
        self.train = [images, labels]
        self.total_images = counter

        self.log.info(
            "Data Read Successfully from {} with total images {} and total Identities {}".format(train_file_path, len(images), self.train_identities))

    def convert_data_to_numpy(self, data):
        self.log.info("Starting to convert the images from python array to numpy array")

        images, labels = self.train
        images = np.array(images)
        labels = np.array(labels)
        self.train_np = (images, labels)

        self.log.info("Successfully converted the images from python array to numpy array")

    def convert_numpy_array_to_hdf5(self, X, Y, X_label, Y_label, path_to_save_hdf5):
        self.log.info("Starting to convert the numpy array X = {}  Y = {} to hdf5".format(X.shape, Y.shape))

        hdf5_file = h5py.File(path_to_save_hdf5, mode='w')
        hdf5_file.create_dataset(X_label, X.shape, np.int8)
        hdf5_file.create_dataset(Y_label, Y.shape, np.int8)
        hdf5_file[self.train_img_hdf5_label][:, ...] = X
        hdf5_file[self.train_label_hdf5_label][:, ...] = Y

        self.log.info(
            "Successfully converted the numpy arrays X = {}  Y = {} to hdf5 with labels \'{}\' and \'{}\' in file {}".format(X.shape, Y.shape,
                                                                                                                             X_label,
                                                                                                                             Y_label,
                                                                                                                             path_to_save_hdf5))

    def create_labeled_unlabeled_sets(self, is_random_in_selected, label_for_labeld_data, label_for_unlabeled_data,
                                      number_of_labelled_image_per_identity=3):

        images, labels = self.train

        labelled_train_X = []
        unlabelled_train_X = []

        labelled_train_Y = []
        unlabelled_train_Y = []

        stats_dict = {}
        for idx, lbl in enumerate(self.original_labels):

            if lbl not in stats_dict.keys():
                stats_dict[lbl] = [(images[idx], labels[idx])]
            else:
                temp_images = stats_dict[lbl]
                temp_images.append((images[idx], labels[idx]))
                stats_dict[lbl] = temp_images

        for key in stats_dict.keys():

            total_images_of_label_to_use = number_of_labelled_image_per_identity
            total_identities = len(stats_dict[key])
            if total_identities < number_of_labelled_image_per_identity:
                self.log.warn("Number of images per label are more than the total images available for this identity {}.".format(key))
                total_images_of_label_to_use = total_identities

            if is_random_in_selected:
                total_images_of_label_to_use = random.randint(2, number_of_labelled_image_per_identity)

            selected_candidates = random.sample(range(1, total_identities), total_images_of_label_to_use)

            self.log.info(
                "Selected {} for labeled selection out of {} and rest for unlabelled on label {}".format(selected_candidates, total_identities, key))

            not_selected = set(list(range(total_identities))) - set(selected_candidates)

            values = stats_dict[key]

            for item in selected_candidates:
                img, lbl = values[item]
                labelled_train_X.append(img)
                labelled_train_Y.append(lbl)

            for item in not_selected:
                img, lbl = values[item]
                unlabelled_train_X.append(img)
                unlabelled_train_Y.append(lbl)

        labelled_train_X = np.array(labelled_train_X)
        labelled_train_Y = np.array(labelled_train_Y)

        unlabelled_train_X = np.array(unlabelled_train_X)
        unlabelled_train_Y = np.array(unlabelled_train_Y)

        self.log.info("Labeled dataset containing X {} and Y {}".format(labelled_train_X.shape, labelled_train_Y.shape))
        self.log.info("Unlabeled dataset containing X {} and Y {}".format(unlabelled_train_X.shape, unlabelled_train_Y.shape))

        self.convert_numpy_array_to_hdf5(labelled_train_X, labelled_train_Y, self.train_img_hdf5_label, self.train_label_hdf5_label,
                                         label_for_labeld_data)
        self.convert_numpy_array_to_hdf5(unlabelled_train_X, unlabelled_train_Y, self.train_img_hdf5_label, self.train_label_hdf5_label,
                                         label_for_unlabeled_data)


if __name__ == '__main__':
    from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

veri = Veri('D:\VeRi', 'image_train', 'image_test', 'image_query', 'name_train_test.txt', 'name_test.txt', 'name_query.txt', (224, 224))

veri.read_train_data(image, preprocess_input, to_categorical)
# veri.convert_data_to_numpy()
# veri.convert_numpy_array_to_hdf5('veri-dataset.hdf5')
veri.create_labeled_unlabeled_sets(False, 'veri-labeled-data.hdf5', 'veri-unlabeled-data.hdf5', 5)
