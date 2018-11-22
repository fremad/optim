from mnist import MNIST
import numpy as np

#TODO shuffle data in here
def ORL_data_import():
    return (np.transpose(np.genfromtxt('./data/orl_data.txt', delimiter="")),
            np.genfromtxt('./data/orl_lbls.txt', delimiter="\t"))

def MNSIT_data_import():
    mndata = MNIST('./data')
    training_images, training_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    training_images = np.asarray(training_images)
    training_labels = np.asarray(training_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    training_images = training_images / 255
    test_images = test_images / 255

    return training_images,test_images, training_labels,test_labels