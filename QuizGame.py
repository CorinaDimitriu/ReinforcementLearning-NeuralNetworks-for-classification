from Loading import load_mnist_dataset, load_cifar10_dataset, load_cifar100_dataset, load_iris_dataset
from NeuralNetwork import apply_nn_without_evaluation
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

warnings.filterwarnings("ignore")

def chose_dataset():
    print('Choose a dataset:')
    print('\t1. MNIST')
    print('\t2. CIFAR-10')
    print('\t3. CIFAR-100')
    print('\t4. Iris')
    print('\t5. Exit')
    print()

    while True:
        dataset_index = input('Enter a number: ')

        if dataset_index == '1':
            return load_mnist_dataset(), ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
        elif dataset_index == '2':
            return load_cifar10_dataset(), ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_index == '3':
            return load_cifar100_dataset(), []
        elif dataset_index == '4':
            return load_iris_dataset(), ['Setosa', 'Versicolor', 'Virginica']
        elif dataset_index == '5':
            exit(0)
        else:
            print('Invalid number')
            print()


def play(dataset, labels):

    trains, (x_test, y_test) = dataset
    attempts = x_test.shape[0]

    predictions = apply_nn_without_evaluation(
        trains, 
        len(labels), 
        epochs= (500 if len(labels) == 3 else 10),
    ).predict([x_test])

    while True:
        test_index = input(f'Enter a image test index (between 0 and {attempts}): ')
    
        if test_index == 'q' or test_index == '':
            break
    
        test_index = int(test_index)
        if test_index < 0 or test_index > attempts:
            print('Invalid index')
            continue
    
        print('\tPrediction: ', labels[np.argmax(predictions[test_index])], end='\n\n') 

        if len(labels) == 3:
            print('Correct answer: ', labels[y_test[test_index]])
            plt.imshow(mpimg.imread('./Iris Images/' + labels[y_test[test_index]] + '.jpg'))
        else:
            plt.imshow(
                x_test[test_index].reshape(
                    x_test[test_index].shape[0], 
                    x_test[test_index].shape[1], 
                    x_test[test_index].shape[2]
                ), 
                cmap=plt.cm.binary
            )
        
        plt.show()

if __name__ == '__main__':
    dataset, labels = chose_dataset()
    play(dataset, labels)
