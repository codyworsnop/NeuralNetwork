import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def calculate_loss(model, X, y):

    index = 0
    correctPredictions = 0

    for sample in X:

        prediction = predict(model, sample)
        label = y[index]

        if (label * prediction > 0):
            correctPredictions += 1

        index += 1

    return correctPredictions / X.Size

def predict(model, x):
    
    #a = x*w1 + b1 
    a = np.matmul(x, model['w1']) + model['b1']

    #h = tanh(a)
    h = np.tanh(a)

    #z = h*w2 + b2 
    z = np.matmul(h, model['w2']) + model['b2']

    #yhat = softmax(z)
    softmax = np.exp(z) / np.sum(np.exp(z))
    
    return np.amax(softmax)

def back_prop(model, y, y_hat, eta):

    #derivative of L with respect to w1
    #x_transpose * (1 - tanh^2(a) * (y_hat - y)* w2_transpose)
    model['w1'] = model['w1'] - eta * ((1 - np.tanh(a)^2) * np.matmul((y_hat - y), np.transpose(model['w2'])))

    #derivative of L with respect to w2
    #h_transpose * (y_hat - y)
    model['w2'] = model['w2'] - eta * (np.matmul(np.transpose()))

    #derivative of L with respect to b1 
    #(1 - tanh^2(a) * (y_hat - y) * w2_transpose)
    model['b1'] = model['b1'] - eta * ()

    #derivative of L with respoect to b2 
    #(y_hat - y)
    model['b2'] = model['b2'] - eta * ()


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
    #initialize weights randomly over the normal distribution, let bias terms init to zero
    model = {
    'w1' : np.random.randn(2, nn_hdim),
    'b1' : 0,
    'w2' : np.random.randn(2, nn_hdim),
    'b2' : 0,
    'h' : 0,
    'a' : 0,
    'z' : 0,
    
    }
    
    for iteration in num_passes:

        #forward propagate and calculate the loss
        loss = calculate_loss(model, X, y)

        #print loss if needed
        if (print_loss and iteration % 1000 == 0):

            print("Current loss value: " + str(loss))

        #back propagate to update the weights 




    return model
    
    
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

plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1, 2, 3, 4]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
   # plot_decision_boundary(lambda x: predict(model, x))


plt.show()
