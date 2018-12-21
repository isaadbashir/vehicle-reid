# Vehicle Re-Identification Gradually Unsupervised Exploiting

## Setup
All our code is implemented in Keras, Tensorflow (Python). Installation instructions are as follows:
```
pip install --user tensorflow-gpu
pip install --user keras
pip install --user sklearn
```

Get the datasets:

- [VeRi](https://github.com/VehicleReId/VeRidataset)
- [VehiceID](https://pkuml.org/resources/pku-vehicleid.html)

and create the train and test list from the dataset and place them in the original dataset folder .

## Create hdf5 files first

The first step is to create the dataset files in hdf5 files e.g.
```Veri(root_dir, train_folder, test_folder, query_folder, train_file, test_file, query_file, input_shape)```
```Veri('D:\VeRi', 'image_train', 'image_test', 'image_query', 'name_train_test.txt', 'name_test.txt', 'name_query.txt', (224, 224))```

then call the method: `read_train_data(image, preprocess_input, to_categorical)` and then `create_labeled_unlabeled_sets(False, 'veri-labeled-data.hdf5', 'veri-unlabeled-data.hdf5', 5)`
`create_labeled_unlabeled_sets(is_random_in_selected, label_for_labeld_data, label_for_unlabeled_data,number_of_labelled_image_per_identity=3)` contains the random functionality

the hdf5 files will be created under the folder `/datasets/veri`

## Running the Gradually Part

To train the method run the `run.py` with required configurations which can be changed including:
`train_loop_strict(number_of_iterations, learning_rate, batch_size, num_of_epochs)` for the policy of assigning new images to every label
`train_loop_loose(number_of_iterations, learning_rate, batch_size, num_of_epochs)` for running the original implementation

## Running the Complete Unsupervised Part

To train the method run the `run-base.py` with required configurations.






