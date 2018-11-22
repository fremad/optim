import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from mnist import MNIST
from data_import import ORL_data_import, MNSIT_data_import

#ORL training data and label import
#(ORL_training_images, ORL_test_images, ORL_training_lbls,ORL_test_lbls) = ORL_data_import()

#MNIST data import
(MNIST_training_images, MNIST_test_images,MNIST_training_labels, MNIST_test_labels) = MNSIT_data_import()

#TODO Add PCA with weights
#Generate PCA
#(MNIST_training_PCA, MNIST_test_PCA) = alg.PCA(MNIST_training_images,2)


# #Nearest centroid classifiers
# lbls = alg.nc_classify(MNIST_training_images,MNIST_test_images,MNIST_training_labels,10)
# print("MNIST NCC classifier errors: ",alg.error_in_percent(MNIST_test_labels,lbls))
#
# ORL_heplbls = alg.nc_classify(ORL_training_images,ORL_test_images,ORL_training_lbls,40)
# print("ORL NCC classifier errors: ",alg.error_in_percent(ORL_test_lbls,ORL_heplbls))
#
# #Nearest neighbor classifier
# MNIST_NN_lbls = alg.nn_classify(MNIST_training_images,MNIST_test_images,MNIST_training_labels)
# print("MNIST NN classifier errors: ",alg.error_in_percent(MNIST_NN_lbls,MNIST_test_labels))
#
# ORL_NN_lbls = alg.nn_classify(ORL_training_images,ORL_test_images,ORL_training_lbls)
# print("ORL NN classifier errors: ", alg.error_in_percent(ORL_NN_lbls,ORL_test_lbls))

#Perceptron with backpropagation

#Perceptron with MSE (least square solution)
MNIST_MSE_lbls = alg.perceptron_MSE_classify(MNIST_training_images,MNIST_test_images,MNIST_training_labels,N_k=10)
print("MNIST MSE classifier error: ", alg.error_in_percent(MNIST_MSE_lbls,MNIST_test_labels))
#tmpdata = alg.PCA(MNIST_training_images,2)


# c = []
#
# for i in range(10):
#     c.append(np.argwhere(MNIST_training_labels==i))
# for c_in in c:
#     plt.plot(tmpdata[1][c_in],tmpdata[0][c_in],'.',color=np.random.rand(3,),markersize=1)
#
# nn_test_labels = alg.nn_classify(MNIST_training_labels,MNIST_test_images,MNIST_training_labels)




def perceptron(learning_rate,class_one,class_two,initial_mean):

    w = initial_mean

    # Input altered with [1 x^t]^t
    aug_c1 = np.ones((class_one.shape[0], class_one.shape[1] + 1))
    aug_c1[:, :-1] = class_one

    aug_c2 = np.ones((class_two.shape[0], class_two.shape[1] + 1))
    aug_c2[:, :-1] = C2

    while(True):

        #print(w)
        c1_guess = w.dot(np.transpose(aug_c1))
        c2_guess = w.dot(np.transpose(aug_c2))

        c1_misslabel = np.argwhere(c1_guess[0] < 0)
        c2_misslabel = np.argwhere(c2_guess[0] >= 0)



        if(c1_misslabel.any() == False):
            if(c2_misslabel.any() == False):
                #print("hep")
                #print(c1_guess,c2_guess,c2_misslabel)
                break


        gradient = np.sum(aug_c1[c1_misslabel],axis=0)-np.sum(aug_c2[c2_misslabel],axis=0)
        #print(gradient)

        w = w + learning_rate * gradient

    return w

#learning_rate = 0.01

C1 = np.array([[-1,0],[0,-1],[-0.5,-0.5],[-1.5,-1.5],[-2,0],[0,-2],[-1,-1.3]])

C2 = np.array([[1,1],[1.3,0.7],[0.7,1.3],[2.5,1],[0,1]])


w = np.array([[0.1, 0.1, 0]])

w = perceptron(0.01,C1,C2,w)


