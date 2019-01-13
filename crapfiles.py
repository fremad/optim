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

# hep = np.zeros((30,784))
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

#nn_test_labels = alg.nn_classify(MNIST_training_lbls,MNIST_test_images,MNIST_training_lbls)



fig, ax = plt.subplots()

for i in range(40):
    b = ORL_training_PCA[np.argwhere(ORL_training_lbls==i).reshape(-1)]
    ax.plot(np.transpose(b)[1],np.transpose(b)[0],'.',color=np.random.rand(3,),markersize=10)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')

fig.tight_layout()
plt.savefig('./PCA_ORL.pdf')
plt.show()

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

# f = plt.figure()
#
# c = []
# for i in range(10):
#     c.append(np.argwhere(MNIST_training_lbls==i))
# for c_in in c:
#     plt.plot(MNIST_training_PCA[c_in][0],MNIST_training_PCA[c_in][1],'.',color=np.random.rand(3,),markersize=1)
# plt.show()



# fig, ax = plt.subplots()
#
# for i in range(10):
#     b = MNIST_training_PCA[np.argwhere(MNIST_test_lbls==i).reshape(-1)]
#     ax.plot(np.transpose(b)[1],np.transpose(b)[0],'.',color=np.random.rand(3,),markersize=1)
#
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
# plt.show()

# fig, ax = plt.subplots()
#
#
#
# for i in range(10):
#     b = MNIST_test_PCA[np.argwhere(lbls==i).reshape(-1)]
#     ax.plot(np.transpose(b)[1],np.transpose(b)[0],'.',color=np.random.rand(3,),markersize=1)
#
# ax.set_xlabel('PCA 1')
# ax.set_ylabel('PCA 2')
# fig.tight_layout()
# plt.savefig('./Perceptron_PCA.pdf')
# plt.show()

# print(error_in_classes(MNIST_test_lbls,MNIST_NCC_lbls,10))
#
# f = plt.figure()
# plt.bar(np.arange(10),error_in_classes(MNIST_test_lbls,MNIST_NCC_lbls,10))
# plt.ylabel('Errors in %')
# plt.xlabel('True class label')
# plt.show()

#
#
# h = plt.figure()
# plt.bar(np.arange(40),error_in_classes(ORL_test_lbls,ORL_NNC_lbls,40))
# plt.ylabel('Errors in %')
# plt.xlabel('True class label')
# plt.show()
# #
# #

# w = perceptron(learning_rate=0.01,Xtrain=MNIST_training_PCA,training_lbls=MNIST_training_lbls, max_iterations=100,N_k=10)
#
# lbls = test_Perceptron(MNIST_test_PCA,w)
#
# print("ERRORS: " ,error_in_percent(lbls,MNIST_test_lbls))
# h = plt.figure()
# plt.bar(np.arange(40),alg.error_in_classes(ORL_test_lbls,ORL_NNC_lbls,40))
# plt.ylabel('Errors in %')
# plt.xlabel('True class label')
# plt.show()