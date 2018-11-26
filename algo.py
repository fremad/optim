import numpy as np
import matplotlib.pyplot as plt

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
    for k in range(n_k.shape[0]):
        plt.imshow(np.reshape(mu_k[k],(30,40)),cmap='gray')
        plt.show()

    lbls = np.zeros(Xtest.shape[0])
    for k in range(Xtest.shape[0]):
        tmp = Xtest[k] - mu_k
        lbls[k] = np.argmin(np.apply_along_axis(np.linalg.norm, 1, tmp))

    return lbls

def nsc_classify(Xtrain,K,Xtest, train_lbls):
    print("not yet implemented")

def nn_classify(Xtrain,Xtest,train_lbls):

    lbls = np.zeros(Xtest.shape[0])

    for i in range(Xtest.shape[0]):
        tmp = np.zeros(Xtrain.shape[0])
        for j in range(Xtrain.shape[0]):
            tmp[j] = np.linalg.norm(np.subtract(Xtest[i], Xtrain[j]))

        lbls[i] = train_lbls[np.argmin(tmp)]

    return lbls

def train_perceptron_backprop(Xtrain, train_lbls, eta):
    print("not implemented")


def perceptron_MSE_classify(training_images, test_images, training_labels, N_k):

    #Pseudo-inverse of X^T
    X_pinv = np.linalg.pinv(np.transpose(training_images))

    w = np.array([])

    for i in range(N_k):
        b = -1 * np.ones(training_images.shape[0])
        b[np.argwhere(training_labels == i)] = 1
        w = np.concatenate((w, np.transpose(X_pinv).dot(b)))
        # answer = np.concatenate((answer,b))

    w = w.reshape((N_k,training_images.shape[1]))

    lbls = np.zeros(test_images.shape[0])
    for i in range(test_images.shape[0]):
        lbls[i] = np.argmax(w.dot(test_images[i]))

    return lbls

def error_in_percent(true_lbls,model_lbls):
    errors = 0.0
    for i in range(true_lbls.shape[0]):
        if(true_lbls[i] != model_lbls[i]):
            errors += 1.0
    return errors/true_lbls.shape[0], errors


def PCA(data,test_data,PCA_components):
    mu_data = np.mean(data, axis=0)
    data_center = data - mu_data


    S_T = np.transpose(data_center).dot(data_center)
    eigenvalues, eigenvectors = np.linalg.eig(S_T)
    eigenvalues = eigenvalues.astype(np.float64)
    eigenvectors = eigenvectors.astype(np.float64)

    ind = np.argpartition(eigenvalues, -PCA_components)[-PCA_components:]

    W = np.array(np.transpose(eigenvectors)[ind])

    test_mu_data = np.mean(test_data, axis=0)
    test_data_center = test_data - test_mu_data

    return np.transpose(W.dot(np.transpose(data_center))), np.transpose(W.dot(np.transpose(test_data_center)))


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
