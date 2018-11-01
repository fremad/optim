import numpy as np
import matplotlib.pyplot as plt
import algo as alg
from mnist import MNIST

from matplotlib.mlab import PCA


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


#Shuffle dataset by creating indexes
s_training = np.arange(training_images.shape[0])
np.random.shuffle(s_training)

s_test = np.arange(test_images.shape[0])
np.random.shuffle(s_test)

#Show MNIST image i
#plt.imshow(np.reshape(training_images[s_training[0]],(28,28)),cmap='gray')
#plt.show()
#print(training_labels[s_training[0]])

#Show ORL image i
# i = 0
# img = X[i].reshape((30,40))
# plt.imshow(img, cmap='gray')
# plt.show()


lbls = alg.nc_classify(training_images,test_images,training_labels,10)

#print(alg.error_in_percent(test_labels,lbls))


#Shuffle dataset by creating indexes
s_ORL = np.arange(ORL_images.shape[0])
np.random.shuffle(s_ORL)

#Center labels to 0
ORL_lbls -= 1


#SPLIT ORL in training/ test

tmp1 = np.split(ORL_images[s_ORL],[350])
tmp2 = np.split(ORL_lbls[s_ORL],[350])

ORL_training_images = tmp1[0]
ORL_test_images = tmp1[1]

ORL_training_lbls = tmp2[0]
ORL_test_lbls = tmp2[1]

ORL_heplbls = alg.nc_classify(ORL_training_images,ORL_test_images,ORL_training_lbls,40)

#print(alg.error_in_percent(ORL_test_lbls,ORL_heplbls))

tmpdata = alg.PCA(training_images,2)


c = []

for i in range(10):
    c.append(np.argwhere(training_labels==i))
for c_in in c:
    plt.plot(tmpdata[0][c_in],tmpdata[1][c_in],'.',color=np.random.rand(3,),markersize=1)


