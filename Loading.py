import numpy as np
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split


def load_mnist_dataset():
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def load_cifar10_dataset():
    num_classes = 10
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def load_cifar100_dataset():
    num_classes = 100
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data("fine")
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")
    # convert class vectors to binary class matrices
    y_train_one_hot = keras.utils.to_categorical(y_train, num_classes)
    y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


def load_iris_dataset():
    num_classes = 3
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    train_labels = keras.utils.to_categorical(y_train, num_classes)
    test_labels = keras.utils.to_categorical(y_test, num_classes)
    normalize_iris(x_train)
    normalize_iris(x_test)
    return (x_train, y_train), (x_test, y_test)


def normalize_iris(dataset):
    for sample in dataset:
        sample[0] = sample[0].astype("float32") / 7.9
        sample[1] = sample[1].astype("float32") / 4.4
        sample[2] = sample[2].astype("float32") / 6.9
        sample[3] = sample[3].astype("float32") / 2.5
