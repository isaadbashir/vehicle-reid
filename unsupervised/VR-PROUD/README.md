# Vehicle Re-Identification Using PROgressive Unsupervised Deep Learning

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

and create the train and test list from the dataset and place them under the `/dataset` respectively.

## Baseline 

1. Contains the `base/train.py` for training the base model which will be used later in the progressive part so simple change the dataset
 variables according to your naming scheme.

## Progressive

1. Rename the above fine-tuned model checkpoint to `0.ckpt` to get started.

2. Create directory `checkpoint` under the folder `VR-PROUD`, and move the original model `0.ckpt` into the `checkpoint`.

3. Modify `PUL/unsupervised.py` and `PUL/evaluate.py` according to the naming scheme of train and test list to train and evaluate.


Please Site the work using this 
```
@InProceedings{RN52,
author="Bashir, Raja Muhammad Saad
and Shahzad, Muhammad
and Fraz, Muhammad Moazam",
title="DUPL-VR: Deep Unsupervised Progressive Learning for Vehicle Re-Identification",
booktitle="Proceedings of the 13th International Symposium on Visual Computing (ISVC)",
year="2018",
pages="286--295"
}
```
