import numpy as np
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator


class TrainGradually(object):
    def __init__(self, net, labelled_data, unlabelled_data, classification_method):
        self.net = net
        self.labelled_data = labelled_data
        self.unlabelled_data = unlabelled_data
        self.train_data = labelled_data
        self.classification_method = classification_method

    def get_image_generator(self):
        return ImageDataGenerator(featurewise_center=False,
                                  samplewise_center=False,
                                  featurewise_std_normalization=False,
                                  samplewise_std_normalization=False,
                                  zca_whitening=False,
                                  rotation_range=20,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.,
                                  zoom_range=0.,
                                  channel_shift_range=0.,
                                  fill_mode='nearest',
                                  cval=0.,
                                  horizontal_flip=False,
                                  vertical_flip=False,
                                  rescale=None,
                                  data_format=K.image_data_format())

    def train_network(self, learning_rate, batch_size, num_of_epochs, itr):
        images, labels = self.train_data
        datagen = self.get_image_generator()
        self.net.compile(optimizer=SGD(lr=learning_rate, momentum=0.9), loss='categorical_crossentropy')
        self.net.fit_generator(datagen.flow(images, labels, batch_size=batch_size), steps_per_epoch=len(images) / batch_size + 1,
                               epochs=num_of_epochs)

        self.net.save('checkpoint/%d.ckpt' % itr)

    def estimate_labels(self):

        if self.classification_method == "classification":
            return self.classify_label_on_unlabelled_data()

        elif self.classification_method == "knn":
            return self.knn_labels_on_unlabelled_data()

    def estimate_labels_strict(self):

        if self.classification_method == "classification":
            return self.classify_label_on_unlabelled_data()

        elif self.classification_method == "knn":
            return self.knn_labels_on_unlabelled_data_strict()

    def classify_label_on_unlabelled_data(self):

        unlabeled_images, unlabeled_labels = self.unlabelled_data
        logits = self.net.predict(unlabeled_images)
        exp_logits = np.exp(logits)
        predict_prob = exp_logits / np.sum(exp_logits, axis=1).reshape((-1, 1))

        pred_label = np.argmax(predict_prob, axis=1)
        pred_score = predict_prob.max(axis=1)
        print("get_Classification_result", predict_prob.shape)

        num_correct_pred = 0
        for idx, p_label in enumerate(pred_label):
            if unlabeled_labels[idx] == p_label:
                num_correct_pred += 1

        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.classification_method, num_correct_pred, pred_label.shape[0], num_correct_pred / pred_label.shape[0]))

        return pred_label, pred_score

    def knn_labels_on_unlabelled_data(self):

        print("Labels")
        labeled_images, labeled_labels = self.labelled_data
        unlabeled_images, unlabeled_labels = self.unlabelled_data

        print("Model")
        feature_net = Model(input=self.net.input, output=self.net.get_layer('avg_pool').output)

        print("Prediction")
        labeled_features = feature_net.predict(labeled_images)
        unlabeled_features = feature_net.predict(unlabeled_images)

        print("Squeezing")
        labeled_features = np.squeeze(labeled_features)
        unlabeled_features = np.squeeze(unlabeled_features)

        scores = np.zeros((unlabeled_features.shape[0]))
        labels = np.zeros((unlabeled_features.shape[0]))

        num_correct_pred = 0

        print("Labelling")
        for idx, u_fea in enumerate(unlabeled_features):

            diffs = labeled_features - u_fea
            dist = np.linalg.norm(diffs, axis=1)
            index_min = np.argmin(dist)
            scores[idx] = - dist[index_min]  # "- dist" : more dist means less score
            lbl = labeled_labels[index_min]
            lbl_argmax = np.argmax(lbl)
            labels[idx] = lbl_argmax  # take the nearest labelled neighbor as the prediction label

            unlbl_argmax = np.argmax(unlabeled_labels[idx])

            # count the correct number of Nearest Neighbor prediction
            if unlbl_argmax == labels[idx]:
                num_correct_pred += 1

        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.classification_method, num_correct_pred, unlabeled_features.shape[0], num_correct_pred / unlabeled_features.shape[0]))

        return labels, scores

    def knn_labels_on_unlabelled_data_strict(self):

        print("Labels")
        labeled_images, labeled_labels = self.labelled_data
        unlabeled_images, unlabeled_labels = self.unlabelled_data

        print("Model")
        feature_net = Model(input=self.net.input, output=self.net.get_layer('avg_pool').output)

        print("Prediction")
        labeled_features = feature_net.predict(labeled_images)
        unlabeled_features = feature_net.predict(unlabeled_images)

        print("Squeezing")
        labeled_features = np.squeeze(labeled_features)
        unlabeled_features = np.squeeze(unlabeled_features)

        scores = np.zeros((labeled_features.shape[0], unlabeled_features.shape[0]))
        # labels = np.zeros((unlabeled_features.shape[0]))

        num_correct_pred = 0

        print("Labelling")
        for idx, l_fea in enumerate(labeled_features):
            diffs = unlabeled_features - l_fea
            dist = np.linalg.norm(diffs, axis=1)
            index_min = np.argmin(dist)
            scores[idx] = - dist  # "- dist" : more dist means less score
            # lbl = labeled_labels[index_min]
            # lbl_argmax = np.argmax(lbl)
            # labels[idx] = lbl_argmax  # take the nearest labelled neighbor as the prediction label

            # unlbl_argmax = np.argmax(unlabeled_labels[idx])

            # count the correct number of Nearest Neighbor prediction
            # if unlbl_argmax == labels[idx]:
            #     num_correct_pred += 1

        # print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
        #     self.classification_method, num_correct_pred, unlabeled_features.shape[0], num_correct_pred / unlabeled_features.shape[0]))

        return scores

    def select_top_data(self, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):
            v[index[i]] = 1
        return v.astype('bool')

    def select_top_data_strict(self, pred_score, nums_to_select):

        v = np.zeros((pred_score.shape))
        index = np.argsort(-pred_score, axis=1)
        for i in range(pred_score.shape[0]):
            for j in range(nums_to_select):
                v[i, index[i, j]] = 1
        return v.astype('bool')

    def generate_new_train_data_strict(self, sel_idx):
        """ generate the next training data """

        labeled_images, labeled_labels = self.labelled_data
        unlabeled_images, unlabeled_labels = self.unlabelled_data

        seletcted_data = []
        seletcted_label = []

        correct, total = 0, 0

        for i in range(sel_idx.shape[0]):
            for j, flag in enumerate(sel_idx[i]):
                if flag:  # if selected
                    seletcted_data.append(unlabeled_images[j])
                    seletcted_label.append(unlabeled_labels[j])
                    total += 1

                    if np.argmax(unlabeled_labels[j]) == np.argmax(labeled_labels[i]):
                        correct += 1

        acc = correct / total

        new_train_data = np.vstack((labeled_images, np.array(seletcted_data)))
        new_train_label = np.vstack((labeled_labels, np.array(seletcted_label)))

        self.train_data = (new_train_data, new_train_label)

        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
            correct, len(seletcted_data), acc, len(new_train_data)))

    def generate_new_train_data(self, sel_idx, pred_y):
        """ generate the next training data """

        labeled_images, labeled_labels = self.labelled_data
        unlabeled_images, unlabeled_labels = self.unlabelled_data

        seletcted_data = []
        seletcted_label = []

        correct, total = 0, 0
        for i, flag in enumerate(sel_idx):
            if flag:  # if selected
                seletcted_data.append(unlabeled_images[i])
                seletcted_label.append(unlabeled_labels[i])

                total += 1
                if np.argmax(unlabeled_labels[i]) == int(pred_y[i]):
                    correct += 1

        acc = correct / total

        new_train_data = np.vstack((labeled_images, np.array(seletcted_data)))
        new_train_label = np.vstack((labeled_labels, np.array(seletcted_label)))

        self.train_data = (new_train_data, new_train_label)

        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
            correct, len(seletcted_data), acc, len(new_train_data)))

    def train_loop_loose(self, number_of_iterations, learning_rate, batch_size, num_of_epochs):

        for itr in range(number_of_iterations):
            print("Staring Training....")
            # self.train_network(learning_rate, batch_size, num_of_epochs, itr)

            print("Estimating Labels....")
            pred_y, pred_score = self.estimate_labels()

            length = len(self.unlabelled_data[0])

            nums_to_select = min(int(length * (itr + 1) * 10 / 100), length)

            # select data
            selected_idx = self.select_top_data(pred_score, nums_to_select)

            # add new data
            self.generate_new_train_data(selected_idx, pred_y)

    def train_loop_strict(self, number_of_iterations, learning_rate, batch_size, num_of_epochs):

        for itr in range(number_of_iterations):
            print("Staring Training....")
            # self.train_network(learning_rate, batch_size, num_of_epochs, itr)

            print("Estimating Labels....")
            pred_score = self.estimate_labels_strict()

            length = len(self.unlabelled_data[0])

            nums_to_select = min(int(length * (itr + 1) * 10 / 100), length)
            print("NUm to select:", nums_to_select)

            # select data
            selected_idx = self.select_top_data_strict(pred_score, nums_to_select)

            # add new data
            self.generate_new_train_data_strict(selected_idx)
