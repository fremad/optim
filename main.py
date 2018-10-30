import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from mnist import MNIST



def nc_classify(Xtrain,Xtest,train_lbls,N_k):

    #Initialize mu an N_k vectors
    mu_k = np.zeros((N_k, Xtrain.shape[1]))
    n_k = np.zeros(N_k)

    #Sum training data into mu_k and count N_k
    for i in range(Xtrain.shape[0]):
        mu_k[train_lbls[i]] += Xtrain[i]
        n_k[train_lbls[i]] += 1

    #divide with N_k to achive mu_ks
    for k in range(n_k.shape[0]):
        mu_k[k] = np.true_divide(mu_k[k], N_k[k])

    lbls = np.zeros(Xtest.shape[0])
    for k in range(Xtest.shape[0]):
        tmp = Xtest[k] - mu_k
        lbls[k] = np.argmin(np.apply_along_axis(np.linalg.norm, 1, tmp))

    return lbls
    # lbls
    # tmp = test_images[s_test[0]] - mu_k
    # k = np.apply_along_axis(np.linalg.norm, 1, tmp)
    # lbl = np.argmin(k)
    # print("not yet implemented")


#ORL training data and label import
ORL_images = np.transpose(np.genfromtxt('./data/orl_data.txt', delimiter=""))
ORL_lbls = np.genfromtxt('./data/orl_lbls.txt', delimiter="\t")

#MNIST data import
mndata = MNIST('./data')
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

training_images = np.asarray(training_images)
training_labels = np.asarray(training_labels)
test_images = np.asarray(test_images)
test_labels = np.asarray(test_labels)

#x = np.array([[3,2],[4,5]])
#y = np.array([2,1])


#Shuffle dataset by creating indexes
s_training = np.arange(training_images.shape[0])
np.random.shuffle(s_training)

s_test = np.arange(test_images.shape[0])
np.random.shuffle(s_test)

#train_shuffle = np.asarray(training_images)
#h = [training_images, training_labels]
#shuf = np.arange(X.shape[0])
#h = np.array()


#Shuffle data

#h = np.apply_along_axis(np.linalg.norm, 1, x)

#train_shuffle = h[0]
#lbls_shuffle = h[1]

#Show MNIST image i
#plt.imshow(np.reshape(training_images[s_training[0]],(28,28)),cmap='gray')
#plt.show()
#print(training_labels[s_training[0]])

#Show ORL image i
# i = 0
# img = X[i].reshape((30,40))
# plt.imshow(img, cmap='gray')
# plt.show()




lbl = nc_classify(training_images,test_images,training_labels,10)

# count = 0
#
# mu_k = np.zeros((10,784))
# N_k = np.zeros(10)
#
# for i in range(s_training.shape[0]):
#     mu_k[training_labels[i]] += training_images[i]
#     N_k[training_labels[i]] += 1
#
# for k in range(N_k.shape[0]):
#     mu_k[k] = np.true_divide(mu_k[k], N_k[k])
#
# tmp = test_images[s_test[0]] - mu_k
# k = np.apply_along_axis(np.linalg.norm, 1,tmp)
# lbl = np.argmin(k)
#
# print('Guess : ',lbl)
# print('Actual : ', test_labels[s_test[0]])