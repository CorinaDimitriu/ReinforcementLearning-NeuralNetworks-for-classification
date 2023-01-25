import os
import time
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
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

acc_nn = None
acc_ppo = None


def draw_lineplot(accuracy_1, accuracy_2):
    epochs = list(range(1, len(accuracy_1[0]) + 1))
    accuracy_1 = zip(*accuracy_1)
    accuracy_dict = {epoch: elem for (epoch, elem) in zip(epochs, accuracy_1)}
    accuracy_dataframe = pd.DataFrame(accuracy_dict)
    df_melted = pd.melt(accuracy_dataframe,
                        value_vars=epochs,
                        value_name='Accuracy_nn',
                        var_name=['Epochs']
                        )
    sns.lineplot(data=df_melted, x="Epochs", y="Accuracy_nn", label="Accuracy_nn", color="blue")
    accuracy_2 = zip(*accuracy_2)
    accuracy_dict = {epoch: elem for (epoch, elem) in zip(epochs, accuracy_2)}
    accuracy_dataframe = pd.DataFrame(accuracy_dict)
    df_melted = pd.melt(accuracy_dataframe,
                        value_vars=epochs,
                        value_name='Accuracy_ppo',
                        var_name=['Epochs']
                        )
    sns.lineplot(data=df_melted, x="Epochs", y="Accuracy_ppo", label="Accuracy_ppo", color="orange")
    plt.legend()
    plt.ylabel("Accuracy")
    plt.show()


def draw_boxplot(accuracy_1, accuracy_2):
    # algos = ["GD", "RL"]
    accuracy_dict = {"GD": accuracy_1[0], "RL": accuracy_2[0]}
    accuracy_dataframe = pd.DataFrame(accuracy_dict)
    df_melted = pd.melt(accuracy_dataframe,
                        value_vars=["GD", "RL"],
                        value_name='Accuracy',
                        var_name=['Algorithm']
                        )
    print(accuracy_dataframe)
    sns.boxplot(data=df_melted, x="Algorithm", y="Accuracy", whis=[5, 95])
    plt.show()


if __name__ == '__main__':
    accuracy_nn = []
    accuracy_ppo = []
    for trial in range(1):
        accuracy_nn.append(apply_nn(load_iris_dataset()[0], load_iris_dataset()[1], 3))
        accuracy_ppo.append(apply_ppo(load_iris_dataset()[0], load_iris_dataset()[1], IrisEnvironment))
        os.system("py Exec.py")
        while not os.path.exists("./Result.txt"):
            time.sleep(1)
        with open("./Result.txt", "rt") as file:
            initialise = file.read()
            exec(initialise)
        accuracy_nn.append(acc_nn)
        accuracy_ppo.append(acc_ppo)
    # draw_lineplot(accuracy_nn, accuracy_ppo)
    draw_boxplot(accuracy_nn, accuracy_ppo)
