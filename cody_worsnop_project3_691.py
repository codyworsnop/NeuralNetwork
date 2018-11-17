import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def calculate_loss(model, X, y):
    thisdict =	{
                    "brand": "Ford",
                    "model": "Mustang",
                    "year": 1964
                }

    a = thisdict["brand"]
    b = 



def predict(model, x):
    
    #a = x*w1 + b1 
    a = np.matmul(x, model['w1']) + model['b1']

    #h = tanh(a)
    h = np.tanh(a)

    #z = h*w2 + b2 
    z = np.matmul(h, model['w2']) + model['b2']

    exo = np.exp(z)
    sumof = sum(exo)

    #yhat = softmax(z)
    softmax = np.exp(z) / np.sum(np.exp(z))
    
    return np.amax(softmax)



def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    a = 3
    
    
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01 

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    #Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    #Z = Z.reshape(xx.shape)

   # plt.contourf(xx, yy, 1)#, cmap=plt.cm.Spectral)
   # plt.scatter(X[:, 0], X[:, 1], c=y)#, cmap=plt.cm.Spectral)


X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
model = {
    'w1' : np.random.randint(low=-10, high=10, size=(2, 2)),
    'b1' : 1,
    'w2' : np.random.randint(low=-10, high=10, size=(2, 2)),
    'b2' : 2
}
predict(model, X[0])
plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1, 2, 3, 4]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer Size %d' % nn_hdim)
  #  model = build_model(X, y, nn_hdim)
   # plot_decision_boundary(lambda x: predict(model, x))


plt.show()
