import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from data_import import ORL_data_import, MNSIT_data_import

from sklearn.cluster import KMeans



#ORL training data and label import
# (ORL_training_images, ORL_test_images, ORL_training_lbls, ORL_test_lbls) = ORL_data_import()

#MNIST data import
(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, MNIST_test_lbls) = MNSIT_data_import()

# #Generate PCA
# (MNIST_training_PCA, MNIST_test_PCA) = alg.PCA(MNIST_training_images,MNIST_test_images,2)
# (ORL_training_PCA, ORL_test_PCA) = alg.PCA(ORL_training_images, ORL_test_images,2)

def augment(data):
    aug_data = np.ones((data.shape[0], data.shape[1] + 1))
    aug_data[:, :-1] = data
    return aug_data


def perceptron(learning_rate,Xtrain,training_lbls, max_iterations):

    #Augment data
    aug_MNIST = augment(Xtrain)

    w_0 = 2*np.random.rand(aug_MNIST.shape[1], 10)-1

    T = -1 *np.ones((10,aug_MNIST.shape[0]))
    for i in range(aug_MNIST.shape[0]):
        T[training_lbls[i]][i] = 1

    f_x = np.zeros((10, aug_MNIST.shape[0]))

    for t in range(max_iterations):

        for i in range(10):
            j = np.transpose(np.transpose(w_0)[i])
            v = j.dot(np.transpose(aug_MNIST))
            h = np.multiply(T[i], v)
            f_x[i] = h

        chi = f_x < 0

        if (chi.any() != 0):
            for i in range(10):
                u = T[i][chi[i,:]]
                p = chi[i]
                lo = np.transpose(aug_MNIST)[:,p]
                kup = np.multiply(u,lo)

                su = np.sum(kup,axis=1)
                #l = T[i][chi[i,:]].dot(aug_MNIST[:,chi[i,:]])

                w_0[:, i] += learning_rate * su
                #h = np.sum(T[i][chi[i,:]].dot(aug_MNIST[:,chi[i,:]]))

                #w_0[:,i] = np.transpose(w_0)[i][:, i] + learning_rate

        else:
            break

    return w_0


w = perceptron(learning_rate=0.01,Xtrain=MNIST_training_images,training_lbls=MNIST_training_lbls, max_iterations=100)

aug_test = augment(MNIST_test_images)
hep = np.transpose(w).dot(np.transpose(aug_test))
hhep = np.transpose(hep)
lbls = np.argmax(hhep,axis=1)

print("ERRORS: " ,alg.error_in_percent(lbls,MNIST_test_lbls))


"""
NEAREST CLASS CENTROID CLASSIFIER
"""

# MNIST_NCC_lbls = alg.nc_classify(MNIST_training_images, MNIST_test_images, MNIST_training_lbls, 10)
# print("MNIST NCC classifier errors: ",alg.error_in_percent(MNIST_test_lbls,MNIST_NCC_lbls))
# #
# ORL_NNC_lbls = alg.nc_classify(ORL_training_images, ORL_test_images, ORL_training_lbls, 40)
# print("ORL NCC classifier errors: ", alg.error_in_percent(ORL_test_lbls, ORL_NNC_lbls))
# #
# #
# MNIST_PCA_NCC_lbls = alg.nc_classify(MNIST_training_PCA, MNIST_test_PCA, MNIST_training_lbls, 10)
# print("MNIST PCA NNC errors", alg.error_in_percent(MNIST_PCA_NCC_lbls,MNIST_test_lbls))
#
# ORL_PCA_NCC_lbls = alg.nc_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls,40)
# print("ORL PCA NNC errors", alg.error_in_percent(ORL_PCA_NCC_lbls,ORL_test_lbls))
#
# """
# NEAREST SUBCLASS CLASSIFIER
# """
#
# #ADD 5 if you dare
# for i in {2,3}:
#     MNIST_NSC_lbls = alg.nsc(MNIST_training_images,MNIST_test_images,MNIST_training_lbls,i,10)
#     print("MNIST NSC K=",i," errors :", alg.error_in_percent(MNIST_NSC_lbls,MNIST_test_lbls))
#
#     ORL_NSC_lbls = alg.nsc(ORL_training_images,ORL_test_images,ORL_training_lbls,i,40)
#     print("ORL NSC K=", i, " errors :", alg.error_in_percent(ORL_test_lbls, ORL_NSC_lbls))
#
#     MNIST_PCA_NSC_lbls = alg.nsc(MNIST_training_PCA, MNIST_test_PCA, MNIST_training_lbls, i, 10)
#     print("MNIST PCA NSC K=", i, " errors :", alg.error_in_percent(MNIST_PCA_NSC_lbls, MNIST_test_lbls))
#
#     ORL_PCA_NSC_lbls = alg.nsc(ORL_training_PCA, ORL_test_PCA, ORL_training_lbls, i, 40)
#     print("ORL PCA NSC K=", i, " errors :", alg.error_in_percent(ORL_test_lbls, ORL_PCA_NSC_lbls))
#
# """
# NEAREST NEIGHBOR CLASSIFIER
# """
#
# # MNIST_NN_lbls = alg.nn_classify(MNIST_training_images,MNIST_test_images,MNIST_training_lbls)
# # print("MNIST NN classifier errors: ", alg.error_in_percent(MNIST_NN_lbls, MNIST_test_lbls))
# #
# # ORL_NN_lbls = alg.nn_classify(ORL_training_images,ORL_test_images,ORL_training_lbls)
# # print("ORL NN classifier errors: ", alg.error_in_percent(ORL_NN_lbls,ORL_test_lbls))
# #
# # MNIST_PCA_NN_lbls = alg.nn_classify(MNIST_training_PCA,MNIST_test_PCA,MNIST_training_lbls)
# # print("MNIST NN classifier errors", alg.error_in_percent(MNIST_PCA_NN_lbls,MNIST_test_lbls))
# #
# # ORL_PCA_NN_lbls = alg.nn_classify(ORL_training_PCA,ORL_test_PCA,ORL_training_lbls)
# # print("ORL PCA NN classifier error : ", alg.error_in_percent(ORL_PCA_NN_lbls,ORL_test_lbls))
#
#
#
# """
# PERCEPTRON WITH BACKPROPAGATION
# """
#
# """
# PERCEPTRON WITH MSE (LEAST SQUARE SOLUTION)
# """
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





# fig, ax = plt.subplots()
#
# for i in range(40):
#     b = ORL_training_PCA[np.argwhere(ORL_training_lbls==i).reshape(-1)]
#     ax.plot(np.transpose(b)[1],np.transpose(b)[0],'.',color=np.random.rand(3,),markersize=1)
#
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
#
# fig.tight_layout()
# plt.savefig('./PCA_ORL.pdf')
# plt.show()

# for i in range(10):
#     c = []
#     c.append(np.argwhere(MNIST_training_lbls==i))
# for c_in in c:
#     plt.plot(MNIST_training_PCA[c_in][0],MNIST_training_PCA[c_in][1],'.',color=np.random.rand(3,),markersize=1)

# # #TODO perceptron neat make neat
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
#         c2_misslabel = np.argwhere(c2_guess[0] > 0)
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
# # learning_rate = 0.01
# #
# # C1 = np.array([[-1,0],[0,-1],[-0.5,-0.5],[-1.5,-1.5],[-2,0],[0,-2],[-1,-1.3]])
# #
# # C2 = np.array([[1,1],[1.3,0.7],[0.7,1.3],[2.5,1],[0,1]])
# #
# #
# # w = np.array([[0.1, 0.1, 0]])
# #
# # w = perceptron(0.01,C1,C2,w,200)
# #
# # X = np.array([[0,0],[1,1],[-1,0],[0.7,-0.2],[-0.2,1.5]])
#

#
# w_k = np.array([])
#
# for i in range(10):
#     C1 = MNIST_training_images[np.argwhere(MNIST_training_lbls == i).reshape(-1)]
#     C2 = MNIST_training_images[np.argwhere(MNIST_training_lbls != i).reshape(-1)]
#     w_k = np.concatenate((w_k, perceptron(0.01, C1, C2, np.random.rand(1, MNIST_training_images.shape[1] + 1), 100)))
#
# w_k = w_k.reshape((785,-1))
#
# aug_MNIST = np.ones((MNIST_test_images.shape[0], MNIST_test_images.shape[1] + 1))
# aug_MNIST[:, :-1] = MNIST_test_images
#
# lbls = np.zeros(MNIST_test_images.shape[0])
#
# w_l = np.zeros((w_k.shape[1],w_k.shape[0]))
# for i in range(w_k.shape[1]):
#     w_l[i] = np.transpose(w_k)[i] / np.linalg.norm(np.transpose(w_k)[i])
#
# #for k in range(MNIST_test_images.shape[0]):
# #    lbls[k] = np.argmax(np.transpose(w_k).dot(np.transpose(aug_MNIST[k])))
#
# for k in range(MNIST_test_images.shape[0]):
#     lbls[k] = np.argmax(w_l.dot(np.transpose(aug_MNIST[k])))
#
# print(alg.error_in_percent(MNIST_test_lbls,lbls))


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
#
#
