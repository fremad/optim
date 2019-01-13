import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def nc_classify(Xtrain,Xtest,train_lbls,N_k):

    #Initialize mu an N_k vectors
    mu_k = np.zeros((N_k, Xtrain.shape[1]))
    n_k = np.zeros(N_k)

    #Sum training data into mu_k and count N_k
    for i in range(Xtrain.shape[0]):
        mu_k[int(train_lbls[i])] += Xtrain[i]
        n_k[int(train_lbls[i])] += 1

    #divide with N_k to achive mu_ks
    for k in range(n_k.shape[0]):
        if(n_k[k] != 0):
            mu_k[k] = np.true_divide(mu_k[k], n_k[k])

    #Uncomment to view the mu_ks as pictures
    # for k in range(n_k.shape[0]):
    #     f = plt.figure()
    ##Change for MNIST
    #     plt.imshow(np.reshape(mu_k[k],(30,40)),cmap='gray')
    #     plt.show()

    #f.savefig('./MNIST_one_mu.pdf', bbox_inches='tight')


    lbls = np.zeros(Xtest.shape[0])
    for k in range(Xtest.shape[0]):
        tmp = Xtest[k] - mu_k
        lbls[k] = np.argmin(np.linalg.norm(tmp,axis=1))

    return lbls

def nsc(Xtrain,Xtest,trainging_labels, K,N_k):

    #Cluster each class in subclasses
    tmp =  np.zeros((K*N_k,Xtrain.shape[1]))
    for i in range(N_k):
        b = Xtrain[np.argwhere(trainging_labels == i).reshape(-1)]
        kmeans = KMeans(n_clusters=K, random_state=0).fit(b)
        for j in range(kmeans.cluster_centers_.shape[0]):
            tmp[kmeans.cluster_centers_.shape[0] * i + j] = kmeans.cluster_centers_[j]

    lbls = np.zeros(Xtest.shape[0])

    #Label according to distance
    for i in range(Xtest.shape[0]):
        tmp_p = Xtest[i] - tmp
        lbls[i] = np.floor(np.argmin(np.apply_along_axis(np.linalg.norm, 1, tmp_p)) / K)

    return lbls

def nn_classify(Xtrain,Xtest,train_lbls):

    lbls = np.zeros(Xtest.shape[0])

    #Find label with least distance
    for i in range(Xtest.shape[0]):
        lbls[i] = train_lbls[np.argmin(np.linalg.norm(Xtrain - Xtest[i],axis=1))]

    return lbls

def perceptron_MSE_classify(training_images, test_images, training_labels, N_k):

    #Pseudo-inverse of X^T
    X_pinv = np.linalg.pinv(np.transpose(training_images))

    w = np.array([])

    #Generate li and create w
    for i in range(N_k):
        b = -1 * np.ones(training_images.shape[0])
        b[np.argwhere(training_labels == i)] = 1
        w = np.concatenate((w, np.transpose(X_pinv).dot(b)))
        # answer = np.concatenate((answer,b))

    w = w.reshape((N_k,training_images.shape[1]))

    #Generate labels
    lbls = np.zeros(test_images.shape[0])
    for i in range(test_images.shape[0]):
        lbls[i] = np.argmax(w.dot(test_images[i]))

    return lbls

def PCA(data,test_data,PCA_components):
    mu_data = np.mean(data, axis=0)
    data_center = data - mu_data

    #Scater matrix
    S_T = np.transpose(data_center).dot(data_center)
    eigenvalues, eigenvectors = np.linalg.eig(S_T)
    eigenvalues = eigenvalues.astype(np.float64)
    eigenvectors = eigenvectors.astype(np.float64)

    #PCA components with higest eigenvalues
    ind = np.argpartition(eigenvalues, -PCA_components)[-PCA_components:]

    W = np.array(np.transpose(eigenvectors)[ind])

    test_mu_data = np.mean(test_data, axis=0)
    test_data_center = test_data - test_mu_data

    print("Norm: ",np.linalg.norm(W))

    return np.transpose(W.dot(np.transpose(data_center))), np.transpose(W.dot(np.transpose(test_data_center)))

def train_perceptron(learning_rate,Xtrain,training_lbls, max_iterations, N_k):

    #Augment data
    aug_Xtrain = augment(Xtrain)

    #Initilize random w
    w_0 = 2*np.random.rand(aug_Xtrain.shape[1], N_k)-1

    T = -1 *np.ones((N_k,aug_Xtrain.shape[0]))
    for i in range(aug_Xtrain.shape[0]):
        T[int(training_lbls[i])][i] = 1

    g_x = np.zeros((N_k, aug_Xtrain.shape[0]))

    #Too avoid non-convergence
    for t in range(max_iterations):

        for i in range(N_k):
            j = np.transpose(np.transpose(w_0)[i])
            v = j.dot(np.transpose(aug_Xtrain))
            h = np.multiply(T[i], v)
            g_x[i] = h

        #Misclassified samples
        missclassified = g_x < 0

        if (missclassified.any() != 0):
            for i in range(N_k):
                u = T[i][missclassified[i,:]]
                p = missclassified[i]
                lo = np.transpose(aug_Xtrain)[:,p]
                kup = np.multiply(u,lo)

                su = np.sum(kup,axis=1)

                w_0[:, i] += learning_rate * su


        else:
            break

    return w_0

def classify_Perceptron(Xtest,w):
    aug_test = augment(Xtest)
    w_lbls = np.transpose(np.transpose(w).dot(np.transpose(aug_test)))
    lbls = np.argmax(w_lbls,axis=1)
    return lbls

"""
HELPER METHODS
"""
#Simple helper
def error_in_percent(true_lbls,model_lbls):
    errors = 0.0
    for i in range(true_lbls.shape[0]):
        if(true_lbls[i] != model_lbls[i]):
            errors += 1.0
    return errors/true_lbls.shape[0], errors, 1- (errors / true_lbls.shape[0])
def error_in_classes(true_lbls, test_lbls,N_k):
    errors = np.zeros(N_k)

    for i in range(test_lbls.shape[0]):
        if(true_lbls[i] != test_lbls[i]):
            k = int(true_lbls[i])
            errors[k] += 1
    errors = errors / np.sum(errors)

    return errors

#Augments [1 X^T]
def augment(data):
    aug_data = np.ones((data.shape[0], data.shape[1] + 1))
    aug_data[:, :-1] = data
    return aug_data

#Not used as it is too slow (WORKS HOWEVER)
def k_means(X,K,initial_mu_k):
    mu_k = initial_mu_k
    while True:
        new_mu_k = np.zeros((mu_k.shape[0], mu_k.shape[1]))
        clusters = []
        for k in range(K):
            clusters.append(np.array([]))

        for l in range(X.shape[0]):
            tmp = np.zeros(mu_k.shape[0])
            for i in range(K):
                # np.subtract(X[0], mu_k[0])
                tmp[i] = np.linalg.norm(np.subtract(X[l], mu_k[i]))

            clusters[np.argmin(tmp)] = np.concatenate((clusters[np.argmin(tmp)], X[l]))
        for j in range(len(clusters)):
            clusters[j] = clusters[j].reshape(-1, X.shape[1])
            # Assign to mu vector

        for f in range(len(clusters)):
            new_mu_k[f] = np.mean(clusters[f], axis=0)

        if (np.linalg.norm(mu_k - new_mu_k) < 0.005):
            mu_k = new_mu_k
            break

        mu_k = new_mu_k

    return mu_k

