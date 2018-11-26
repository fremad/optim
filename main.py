import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from data_import import ORL_data_import, MNSIT_data_import

from sklearn.cluster import KMeans



#ORL training data and label import
(ORL_training_images, ORL_test_images, ORL_training_lbls, ORL_test_lbls) = ORL_data_import()

#MNIST data import
(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, MNIST_test_lbls) = MNSIT_data_import()

#Generate PCA
(MNIST_training_PCA, MNIST_test_PCA) = alg.PCA(MNIST_training_images,MNIST_test_images,2)
#(ORL_training_PCA, ORL_test_PCA) = alg.PCA(ORL_training_images, ORL_test_images,2)



"""
NEAREST CLASS CENTROID CLASSIFIER
"""

#MNIST_NCC_lbls = alg.nc_classify(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, 10)
#print("MNIST NCC classifier errors: ",alg.error_in_percent(MNIST_test_lbls,MNIST_NCC_lbls))
#
ORL_NNC_lbls = alg.nc_classify(ORL_training_images, ORL_test_images, ORL_training_lbls, 40)
print("ORL NCC classifier errors: ", alg.error_in_percent(ORL_test_lbls, ORL_NNC_lbls))
#
#
# MNIST_PCA_NCC_lbls = alg.nc_classify(MNIST_training_PCA, MNIST_test_PCA, MNIST_training_lbls, 10)
# print("MNIST PCA NNC errors", alg.error_in_percent(MNIST_PCA_NCC_lbls,MNIST_test_labels))
#
# ORL_PCA_NCC_lbls = alg.nc_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls,40)
# print("ORL PCA NNC errors", alg.error_in_percent(ORL_PCA_NCC_lbls,ORL_test_lbls))



"""
NEAREST NEIGHBOR CLASSIFIER
"""

# MNIST_NN_lbls = alg.nn_classify(MNIST_training_images,MNIST_test_images,MNIST_training_lbls)
# print("MNIST NN classifier errors: ", alg.error_in_percent(MNIST_NN_lbls, MNIST_test_lbls))
#
# ORL_NN_lbls = alg.nn_classify(ORL_training_images,ORL_test_images,ORL_training_lbls)
# print("ORL NN classifier errors: ", alg.error_in_percent(ORL_NN_lbls,ORL_test_lbls))

# MNIST_PCA_NN_lbls = alg.nn_classify(MNIST_training_PCA,MNIST_test_PCA,MNIST_training_lbls)
# print("MNIST NN classifier errors", alg.error_in_percent(MNIST_PCA_NN_lbls,MNIST_test_lbls))
#
# ORL_PCA_NN_lbls = alg.nn_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls)
# print("ORL PCA NN classifier error : ", alg.error_in_percent(ORL_PCA_NN_lbls,ORL_test_lbls))

"""
NEAREST SUBCLASS CLASSIFIER
"""

# hep = np.zeros((30,784))
# b = np.array([])
# for i in range(10):
#     b = MNIST_training_images[np.argwhere(MNIST_training_lbls == i).reshape(-1)]
#     kmeans = KMeans(n_clusters=3, random_state=0).fit(b)
#     for j in range(kmeans.cluster_centers_.shape[0]):
#         hep[kmeans.cluster_centers_.shape[0]*i+j] = kmeans.cluster_centers_[j]
#
# lbls = np.zeros(MNIST_test_images.shape[0])
#
# for i in range(MNIST_test_lbls.shape[0]):
#     tmp = MNIST_test_images[i]-hep
#     lbls[i] = np.floor(np.argmin(np.apply_along_axis(np.linalg.norm, 1, tmp)) / 3)
#
# alg.error_in_percent(lbls,MNIST_test_lbls)

"""
PERCEPTRON WITH BACKPROPAGATION
"""

"""
PERCEPTRON WITH MSE (LEAST SQUARE SOLUTION)
"""


# MNIST_MSE_lbls = alg.perceptron_MSE_classify(MNIST_training_images,MNIST_test_images,MNIST_training_lbls,N_k=10)
# print("MNIST MSE classifier error: ", alg.error_in_percent(MNIST_MSE_lbls,MNIST_test_lbls))
#
# ORL_MSE_lbls = alg.perceptron_MSE_classify(ORL_training_images,ORL_test_images,ORL_training_lbls,N_k=40)
# print("ORL MSE classifier error: ", alg.error_in_percent(ORL_MSE_lbls,ORL_test_lbls))
#
# MNIST_PCA_MSE_lbls = alg.perceptron_MSE_classify(MNIST_training_PCA,MNIST_test_PCA,MNIST_training_lbls,N_k=10)
# print("MNIST PCA MSE classifier error: ", alg.error_in_percent(MNIST_PCA_MSE_lbls,MNIST_test_lbls))
#
# ORL_PCA_MSE_lbls = alg.perceptron_MSE_classify(ORL_training_PCA, ORL_test_PCA,ORL_training_lbls,N_k=40)
# print("ORL PCA MSE classifier error: ", alg.error_in_percent(ORL_PCA_MSE_lbls,ORL_test_lbls))



#tmpdata = alg.PCA(MNIST_training_images,2)


# c = []
#
# for i in range(10):
#     c.append(np.argwhere(MNIST_training_labels==i))
# for c_in in c:
#     plt.plot(tmpdata[1][c_in],tmpdata[0][c_in],'.',color=np.random.rand(3,),markersize=1)
#
# nn_test_labels = alg.nn_classify(MNIST_training_labels,MNIST_test_images,MNIST_training_labels)



# #TODO perceptron neat make neat
# def perceptron(learning_rate,class_one,class_two,initial_mean,max_iterations):
#
#     w = initial_mean
#
#     # Input altered with [1 x^t]^t
#     aug_c1 = np.ones((class_one.shape[0], class_one.shape[1] +1))
#     aug_c1[:, :-1] = class_one
#
#     aug_c2 = np.ones((class_two.shape[0], class_two.shape[1] +1))
#     aug_c2[:, :-1] = class_two
#
#     for i in range(max_iterations):
#
#         #print(w)
#         c1_guess = w.dot(np.transpose(aug_c1))
#         c2_guess = w.dot(np.transpose(aug_c2))
#
#         c1_misslabel = np.argwhere(c1_guess[0] < 0)
#         c2_misslabel = np.argwhere(c2_guess[0] >= 0)
#
#
#         if(c1_misslabel.any() == False):
#             if(c2_misslabel.any() == False):
#                 #print("hep")
#                 #print(c1_guess,c2_guess,c2_misslabel)
#                 break
#
#         #print(np.sum(aug_c1[c1_misslabel], axis=0).shape)
#         gradient = np.sum(aug_c1[c1_misslabel],axis=0)-np.sum(aug_c2[c2_misslabel],axis=0)
#         #print(gradient.shape)
#
#
#
#         w = w + learning_rate * gradient
#
#     return w[0]
#
# #learning_rate = 0.01
#
# # C1 = np.array([[-1,0],[0,-1],[-0.5,-0.5],[-1.5,-1.5],[-2,0],[0,-2],[-1,-1.3]])
# #
# # C2 = np.array([[1,1],[1.3,0.7],[0.7,1.3],[2.5,1],[0,1]])
# #
# #
# # w = np.array([[0.1, 0.1, 0]])
# #
# # w = perceptron(0.01,C1,C2,w,200)
#
# w_k = np.array([])
#
# # for i in range(40):
# #     C1 = ORL_training_images[np.argwhere(ORL_training_lbls == i).reshape(-1)]
# #     C2 = ORL_training_images[np.argwhere(ORL_training_lbls != i).reshape(-1)]
# #
# #     w_k = np.concatenate((w_k,perceptron(0.001,C1,C2,np.random.rand(1,ORL_training_images.shape[1]+1),50000)))
# #
# # w_k = w_k.reshape((1201,-1))
# #
# #
# # aug_ORL = np.ones((ORL_test_images.shape[0], ORL_test_images.shape[1] + 1))
# # aug_ORL[:, :-1] = ORL_test_images
# #
# # lbls = np.zeros(ORL_test_images.shape[0])
# #
# # for k in range(ORL_test_images.shape[0]):
# #     lbls[k] = np.argmax(np.transpose(w_k).dot(aug_ORL[k]))
# #
# # print(alg.error_in_percent(ORL_test_lbls,lbls))
#
# for i in range(10):
#     C1 = MNIST_training_images[np.argwhere(MNIST_training_labels == i).reshape(-1)]
#     C2 = MNIST_training_images[np.argwhere(MNIST_training_labels != i).reshape(-1)]
#     print(i)
#     w_k = np.concatenate((w_k,perceptron(0.001,C1,C2,np.random.rand(1,MNIST_training_images.shape[1]+1),100)))
#
# w_k = w_k.reshape((785,-1))
#
#
# aug_MNIST = np.ones((MNIST_test_images.shape[0], MNIST_test_images.shape[1] + 1))
# aug_MNIST[:, :-1] = MNIST_test_images
#
# lbls = np.zeros(MNIST_test_images.shape[0])
#
# for k in range(MNIST_test_images.shape[0]):
#     lbls[k] = np.argmax(np.transpose(w_k).dot(aug_MNIST[k]))
#
# print(alg.error_in_percent(MNIST_test_labels,lbls))