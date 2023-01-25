# Sensitive cases
# 4356

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time


def create_model(classes):
    model = tf.keras.models.Sequential()  # a feed-forward model

    model.add(tf.keras.layers.Flatten())  # input layer
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))  # hidden layer #1
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.sigmoid))  # hidden layer #2
    model.add(tf.keras.layers.Dense(classes, activation=tf.nn.softmax))  # output layer

    model.compile(optimizer='adam',
                  # Optimizer that implements the Adam algorithm. Adam optimization is a stochastic gradient descent
                  # method that is based on adaptive estimation of first-order and second-order moments. According to
                  # Kingma et al., 2014, the method is "computationally efficient, has little memory requirement,
                  # invariant to diagonal rescaling of gradients, and is well suited for problems that are large in
                  # terms of data/parameters".
                  loss='sparse_categorical_crossentropy',
                  # sparse_categorical_crossentropy is used for non-one-hot encoded labels
                  metrics=['accuracy']
                  # what to track
                  )

    return model


def apply_nn(training_set, testing_set, classes):
    accuracies = []
    x_train, y_train = training_set
    nn_model = create_model(classes)
    for epoch in range(50):
        nn_model.fit(x_train, y_train, epochs=1)  # train the model for 1 epoch
        accuracies.append(evaluate_nn(testing_set, nn_model))
    nn_model.save("./Model_NN")
    return accuracies


def evaluate_nn(testing_set, trained_model):
    x_test, y_test = testing_set
    predictions = trained_model.predict([x_test])
    # x_test[i] -> predictions[i] = [y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9]
    #           -> where y_i is the probability that the image is a digit i
    #           -> the highest probability is the prediction of the model

    attempts = x_test.shape[0]
    attempted, correctly_predicted = attempts, 0
    for i in range(attempts):
        if np.argmax(predictions[i]) == y_test[i]:
            correctly_predicted += 1

    # report = open("Accuracy.txt", "a")
    # print('\nFinished validation with %f accuracy\n'
    #       % ((float(correctly_predicted) / attempted) * 100), file=report)
    return (float(correctly_predicted) / attempted) * 100

