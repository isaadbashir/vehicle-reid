import glob
import os

import h5py
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

hdf5_path = '../dataset/VeRi/hdf5/veri-dataset.hdf5'  # address to where you want to save the hdf5 file
veri_dataset_path = 'D:/VeRi/image_train/'
veri_data_file = '../dataset/VeRi/name_train.txt'
total_labels = 770


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


# read addresses and labels from the 'train' folder
addrs = glob.glob(os.path.join(veri_dataset_path, '*.jpg'))

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow

# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(addrs), 3, 224, 224)
elif data_order == 'tf':
    train_shape = (len(addrs), 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("train_labels", (len(addrs), total_labels), np.int8)

printProgressBar(0, len(addrs), prefix='Progress:', suffix='Complete', length=50)

i = 0
# load data

label = []
with open(veri_data_file, 'r') as f:
    for line in f:
        line = line.strip()

        img = line
        lbl = int(img.split("_")[0])

        imagePath = os.path.join(veri_dataset_path, img)

        img = image.load_img(imagePath, target_size=[224, 224])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        img = np.array(img)

        # if the data order is Theano, axis orders should change
        if data_order == 'th':
            img = np.rollaxis(img, 2)
        # save the image and calculate the mean so far

        hdf5_file["train_img"][i, ...] = img[None]
        label.append(lbl)

        printProgressBar(i + 1, len(addrs), prefix='Progress:', suffix='Complete', length=50)
        i = i + 1

i = i - 1
label = to_categorical(lbl)
hdf5_file["train_labels"][i, ...] = label

hdf5_file.close()
