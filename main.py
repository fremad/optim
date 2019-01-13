import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from data_import import ORL_data_import, MNSIT_data_import


#ORL training data and label import
(ORL_training_images, ORL_test_images, ORL_training_lbls, ORL_test_lbls) = ORL_data_import()

#MNIST data import
(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, MNIST_test_lbls) = MNSIT_data_import()

# #Generate PCA
(MNIST_training_PCA, MNIST_test_PCA) = alg.PCA(MNIST_training_images,MNIST_test_images,2)
(ORL_training_PCA, ORL_test_PCA) = alg.PCA(ORL_training_images, ORL_test_images,2)


"""
NEAREST CLASS CENTROID CLASSIFIER
"""

MNIST_NCC_lbls = alg.nc_classify(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, 10)
print("MNIST NCC classifier errors: ",alg.error_in_percent(MNIST_test_lbls,MNIST_NCC_lbls))

ORL_NNC_lbls = alg.nc_classify(ORL_training_images, ORL_test_images, ORL_training_lbls, 40)
print("ORL NCC classifier errors: ", alg.error_in_percent(ORL_test_lbls, ORL_NNC_lbls))

MNIST_PCA_NCC_lbls = alg.nc_classify(MNIST_training_PCA, MNIST_test_PCA, MNIST_training_lbls, 10)
print("MNIST PCA NNC errors", alg.error_in_percent(MNIST_PCA_NCC_lbls,MNIST_test_lbls))

ORL_PCA_NCC_lbls = alg.nc_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls,40)
print("ORL PCA NNC errors", alg.error_in_percent(ORL_PCA_NCC_lbls,ORL_test_lbls))



"""
NEAREST SUBCLASS CLASSIFIER
"""
#ADD 5 if you dare
for i in {2,3,5}:
    MNIST_NSC_lbls = alg.nsc(MNIST_training_images,MNIST_test_images,MNIST_training_lbls,i,10)
    print("MNIST NSC K=",i," errors :", alg.error_in_percent(MNIST_NSC_lbls,MNIST_test_lbls))

    ORL_NSC_lbls = alg.nsc(ORL_training_images,ORL_test_images,ORL_training_lbls,i,40)
    print("ORL NSC K=", i, " errors :", alg.error_in_percent(ORL_test_lbls, ORL_NSC_lbls))

    MNIST_PCA_NSC_lbls = alg.nsc(MNIST_training_PCA, MNIST_test_PCA, MNIST_training_lbls, i, 10)
    print("MNIST PCA NSC K=", i, " errors :", alg.error_in_percent(MNIST_PCA_NSC_lbls, MNIST_test_lbls))

    ORL_PCA_NSC_lbls = alg.nsc(ORL_training_PCA, ORL_test_PCA, ORL_training_lbls, i, 40)
    print("ORL PCA NSC K=", i, " errors :", alg.error_in_percent(ORL_test_lbls, ORL_PCA_NSC_lbls))


"""
NEAREST NEIGHBOR CLASSIFIER
"""

MNIST_NN_lbls = alg.nn_classify(MNIST_training_images,MNIST_test_images,MNIST_training_lbls)
print("MNIST NN classifier errors: ", alg.error_in_percent(MNIST_NN_lbls, MNIST_test_lbls))

ORL_NN_lbls = alg.nn_classify(ORL_training_images,ORL_test_images,ORL_training_lbls)
print("ORL NN classifier errors: ", alg.error_in_percent(ORL_NN_lbls,ORL_test_lbls))

MNIST_PCA_NN_lbls = alg.nn_classify(MNIST_training_PCA,MNIST_test_PCA,MNIST_training_lbls)
print("MNIST NN classifier errors", alg.error_in_percent(MNIST_PCA_NN_lbls,MNIST_test_lbls))

ORL_PCA_NN_lbls = alg.nn_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls)
print("ORL PCA NN classifier error : ", alg.error_in_percent(ORL_PCA_NN_lbls,ORL_test_lbls))


"""
PERCEPTRON WITH BACKPROPAGATION
"""

MNIST_backprop_lbls = alg.classify_Perceptron(MNIST_test_images,alg.train_perceptron(learning_rate=0.01,Xtrain=MNIST_training_images,training_lbls=MNIST_training_lbls, max_iterations=100,N_k=10))
print("MNIST backprop classifier errors", alg.error_in_percent(MNIST_backprop_lbls,MNIST_test_lbls))

ORL_backprop_lbls = alg.classify_Perceptron(ORL_test_images,alg.train_perceptron(learning_rate=0.01,Xtrain=ORL_training_images,training_lbls=ORL_training_lbls, max_iterations=100,N_k=40))
print("ORL backprop classifier errors", alg.error_in_percent(ORL_backprop_lbls,ORL_test_lbls))

MNIST_PCA_backprop_lbls = alg.classify_Perceptron(MNIST_test_PCA,alg.train_perceptron(learning_rate=0.01,Xtrain=MNIST_training_PCA,training_lbls=MNIST_training_lbls, max_iterations=100,N_k=10))
print("MNIST PCA backprop classifier errors", alg.error_in_percent(MNIST_PCA_backprop_lbls,MNIST_test_lbls))

ORL_PCA_backprop_lbls = alg.classify_Perceptron(ORL_test_PCA,alg.train_perceptron(learning_rate=0.01,Xtrain=ORL_training_PCA,training_lbls=ORL_training_lbls, max_iterations=100,N_k=40))
print("ORL PCA backprop classifier errors", alg.error_in_percent(ORL_PCA_backprop_lbls,ORL_test_lbls))


"""
PERCEPTRON WITH MSE (LEAST SQUARE SOLUTION)
"""
MNIST_MSE_lbls = alg.perceptron_MSE_classify(MNIST_training_images,MNIST_test_images,MNIST_training_lbls,N_k=10)
print("MNIST MSE classifier error: ", alg.error_in_percent(MNIST_MSE_lbls,MNIST_test_lbls))

ORL_MSE_lbls = alg.perceptron_MSE_classify(ORL_training_images,ORL_test_images,ORL_training_lbls,N_k=40)
print("ORL MSE classifier error: ", alg.error_in_percent(ORL_MSE_lbls,ORL_test_lbls))

MNIST_PCA_MSE_lbls = alg.perceptron_MSE_classify(MNIST_training_PCA,MNIST_test_PCA,MNIST_training_lbls,N_k=10)
print("MNIST PCA MSE classifier error: ", alg.error_in_percent(MNIST_PCA_MSE_lbls,MNIST_test_lbls))

ORL_PCA_MSE_lbls = alg.perceptron_MSE_classify(ORL_training_PCA, ORL_test_PCA,ORL_training_lbls,N_k=40)
print("ORL PCA MSE classifier error: ", alg.error_in_percent(ORL_PCA_MSE_lbls,ORL_test_lbls))
