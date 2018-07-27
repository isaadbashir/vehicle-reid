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



<!--If you find this code useful, consider citing our work:
```
@article{fan17unsupervised,
  author    = {Hehe Fan and Liang Zheng and Yi Yang},
  title     = {Unsupervised Person Re-identification: Clustering and Fine-tuning},
  journal   = {arXiv preprint arXiv:1705.10444},
  year      = {2017}
}
```
-->
