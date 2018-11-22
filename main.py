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


