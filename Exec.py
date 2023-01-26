from Loading import load_mnist_dataset
from Loading import load_cifar10_dataset
from Loading import load_cifar100_dataset
from Loading import load_iris_dataset
from MnistEnvironment import MnistEnvironment
from IrisEnvironment import IrisEnvironment
from Cifar10Environment import Cifar10Environment
from Cifar100Environment import Cifar100Environment
from NeuralNetwork import apply_nn
from Ppo import apply_ppo


with open("Result.txt", "wt") as file:
    file.write(("acc_nn = " + str(apply_nn(load_cifar10_dataset()[0], load_cifar10_dataset()[1], 10)) + "\n" +
                "acc_ppo = " + str(apply_ppo(load_cifar10_dataset()[0], load_cifar10_dataset()[1],
                                             Cifar10Environment))))

