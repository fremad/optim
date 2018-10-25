import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from mnist import MNIST


def nc_classify(Xtrain,Xtest,train_lbls):
    print("not yet implemented")


#ORL training data and label import
X = np.transpose(np.genfromtxt('./data/orl_data.txt', delimiter=""))
lbls = np.genfromtxt('./data/orl_lbls.txt', delimiter="\t")

#MNIST data import
mndata = MNIST('./data')
training_images, training_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


x = np.array([[3,2],[4,5]])
y = np.array([2,1])


h = [training_images, training_labels]
#Show MNIST image i
plt.imshow(np.reshape(training_images[1],(28,28)),cmap='gray')
plt.show()

#Show ORL image i
# i = 0
# img = X[i].reshape((30,40))
# plt.imshow(img, cmap='gray')
# plt.show()