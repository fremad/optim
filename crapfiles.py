# #Shuffle dataset by creating indexes
# s_training = np.arange(training_images.shape[0])
# np.random.shuffle(s_training)
#
# s_test = np.arange(test_images.shape[0])
# np.random.shuffle(s_test)


#Show MNIST image i
#plt.imshow(np.reshape(training_images[s_training[0]],(28,28)),cmap='gray')
#plt.show()
#print(training_labels[s_training[0]])



#Show ORL image i
# i = 0
# img = X[i].reshape((30,40))
# plt.imshow(img, cmap='gray')
# plt.show()

# #Shuffle dataset by creating indexes
# s_ORL = np.arange(ORL_images.shape[0])
# np.random.shuffle(s_ORL)
#
# #Center labels to 0
# ORL_lbls -= 1
#
#
# #SPLIT ORL in training/ test
#
# tmp1 = np.split(ORL_images[s_ORL],[350])
# tmp2 = np.split(ORL_lbls[s_ORL],[350])
#
# ORL_training_images = tmp1[0]
# ORL_test_images = tmp1[1]
#
# ORL_training_lbls = tmp2[0]
# ORL_test_lbls = tmp2[1]

#(ORL_images, ORL_lbls) = ORL_data_import()





#USELESS JUNK!!!!!

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


### NEW SECTION !!!!

# mean = np.apply_along_axis(np.sum,0,X)/X.shape[0]
#
# X_mean = X-mean
#
# S_T = np.transpose(X_mean).dot(X_mean)
#
# eigenvalues, eigenvectors = np.linalg.eig(S_T)