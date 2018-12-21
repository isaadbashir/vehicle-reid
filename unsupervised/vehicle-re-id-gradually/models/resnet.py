from keras.applications import ResNet50
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model


class ResNet(object):
    def __init__(self, number_of_classes, weights, include_top, input_shape):
        self.number_of_classes = number_of_classes
        self.weights = weights
        self.include_top = include_top
        self.input_shape = input_shape

    def get_resnet_model(self):
        base_model = ResNet50(weights=self.weights, include_top=self.include_top, input_tensor=Input(shape=self.input_shape))

        x = base_model.output
        x = Flatten(name='flatten')(x)
        x = Dropout(0.5)(x)
        x = Dense(self.number_of_classes, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
        net = Model(input=base_model.input, output=x)

        for layer in net.layers:
            layer.trainable = True

        return net
