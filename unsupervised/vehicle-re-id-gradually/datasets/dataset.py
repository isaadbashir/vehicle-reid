import logging
from abc import abstractmethod

from utils.utils import Utils


class Dataset(object):
    def __init__(self, root_dir, train_folder, test_folder, query_folder, train_file, test_file, query_file,
                 input_shape):
        self.root_dir = root_dir
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.query_folder = query_folder
        self.train_file = train_file
        self.test_file = test_file
        self.query_file = query_file
        self.train_img_hdf5_label = "train_images"
        self.train_label_hdf5_label = "train_labels"
        self.input_shape = input_shape
        self.train_identities = 0
        self.total_images = 0
        self.number_of_channels = 3
        self.train = []
        self.test = []
        self.query = []

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('vehicle-reid.log')

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.log.addHandler(console_handler)
        self.log.addHandler(file_handler)

    @property
    def get_train_dir(self):
        return Utils.join_paths(self.root_dir, self.train_folder)

    @property
    def get_test_dir(self):
        return Utils.join_paths(self.root_dir, self.test_folder)

    @property
    def get_query_dir(self):
        return Utils.join_paths(self.root_dir, self.query_folder)

    @abstractmethod
    def read_train_data(self):
        pass

    @abstractmethod
    def read_test_data(self):
        pass

    @abstractmethod
    def read_query_data(self):
        pass
