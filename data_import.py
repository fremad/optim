from mnist import MNIST
import numpy as np


def ORL_data_import():
    (ORL_images, ORL_lbls) = (np.transpose(np.genfromtxt('./data/orl_data.txt', delimiter="")),
            np.genfromtxt('./data/orl_lbls.txt', delimiter="\t"))

    # Put dataset in order
    ORL_training = np.zeros((ORL_images.shape[0],ORL_images.shape[1]))
    ORL_training_lbls = np.zeros(ORL_lbls.shape[0])
    for j in range(10):
        for i in range(40):
            ORL_training[40*j+i] = ORL_images[10*i+j]
            ORL_training_lbls[40 * j + i] = ORL_lbls[10 * i + j]
    # Shuffle dataset by creating indexes
    s_ORL = np.arange(ORL_images.shape[0])
    np.random.shuffle(s_ORL)

    # Center labels to 0
    ORL_lbls -= 1


    #Split dataset to test and training
    tmp1 = np.split(ORL_images[s_ORL], [280])
    tmp2 = np.split(ORL_lbls[s_ORL], [280])

    ORL_training_images = tmp1[0]
    ORL_test_images = tmp1[1]

    ORL_training_lbls = tmp2[0]
    ORL_test_lbls = tmp2[1]

    return ORL_training_images, ORL_test_images, ORL_training_lbls, ORL_test_lbls

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