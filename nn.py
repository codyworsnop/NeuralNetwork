import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons
import time

def calculate_loss(model, X, y):

    #L(y, y_hat) = - 1/N * sum(y_n * log(y_hat_n))
    lossSum = 0.0

    y_hat = predict_many(model, X)

    for n in range(y.shape[0]):
        lossSum += np.log(y_hat[n][y[n]])

    return ((-1/y.shape[0]) * lossSum)

def forward_prop(model, x):

    #a = x*w1 + b1 
    a = np.dot(x, model['w1']) + model['b1']

    #h = tanh(a)
    h = np.tanh(a)

    #z = h*w2 + b2 
    z = np.dot(h, model['w2']) + model['b2']

    #yhat = softmax(z)
    softmax = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    return (a, h, z, softmax)

def predict(model, x):

    a, h, z, softmax = forward_prop(model, x)

    return np.argmax(softmax, axis=1)

def predict_many(model, X, outputSize=2):

    a, h, z, softmax = forward_prop(model, X)

    #update model
    model['a'].append(a)
    model['h'].append(h)
    model['z'].append(z)

    return softmax

def diff_y(y_hat, y):

    diff_y = np.array(y_hat)
    for iteration in range(0, y.shape[0]):

        diff_y[iteration][y[iteration]] -= 1 
    
    return diff_y

def grad_desc(model, X, y, y_hat, nn_hdim):

    eta = 0.01

    y_difference = diff_y(y_hat, y)
    tan_func = (1 - (np.tanh(model['a'][0]) ** 2))

    #derivative of L with respect to w2 
    #h_transpose * (y_hat - y)
    model['w2'] = model['w2'] - eta * np.dot(np.transpose(model['h'][0]), y_difference)

    #derivative of L with respect to b2 
    #y_hat - y
    model['b2'] = model['b2'] - eta * np.sum(diff_y(y_hat, y), axis=0, keepdims=True)
     
    #derivative of L with respect to w1
    #x_transpose * (1 - tanh^2(a) * (y_hat - y) * w2_transpose)
    model['w1'] = model['w1'] - eta * np.dot(np.transpose(X), tan_func * np.dot(y_difference, np.transpose(model['w2'])))

    #derivative of L with respect to b1
    #(1 - tanh^2(a) * (y_hat - y) * w2_transpose)
    model['b1'] = model['b1'] - eta * np.sum(tan_func * np.dot(y_difference, np.transpose(model['w2'])), axis=0)

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
    #initialize weights randomly over the normal distribution, let bias terms init to zero
    model = {
        'w1' : np.random.randn(2, nn_hdim),
        'b1' : np.zeros(shape=(1, nn_hdim)),
        'w2' : np.random.randn(nn_hdim, 2),
        'b2' : np.zeros(shape=(1, 2)),
        'h' : [],
        'a' : [],
        'z' : [],  
    }
    
    for iteration in range(num_passes):

        #forward propagate
        predictions = predict_many(model, X)

        #print loss if needed
        if (print_loss and iteration % 1000 == 0 and iteration != 0):

            #calculate the loss
            loss = calculate_loss(model, X, y)
            print("Current loss value: " + str(loss))

        #back propagate to update the weights 
        grad_desc(model, X, y, predictions, nn_hdim)

        #reset a, h, z
        model['h'] = []
        model['a'] = []
        model['z'] = []

    return model

def build_model_691(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
    #initialize weights randomly over the normal distribution, let bias terms init to zero
    model = {
        'w1' : np.random.randn(2, nn_hdim),
        'b1' : np.zeros(shape=(1, nn_hdim)),
        'w2' : np.random.randn(nn_hdim, 3),
        'b2' : np.zeros(shape=(1, 3)),
        'h' : [],
        'a' : [],
        'z' : [],  
    }

    for iteration in range(num_passes):

        #forward propagate
        predictions = predict_many(model, X, 3)

        #print loss if needed
        if (print_loss and iteration % 1000 == 0 and iteration != 0):

            #calculate the loss
            loss = calculate_loss(model, X, y)
            print("Current loss value: " + str(loss))

        #back propagate to update the weights 
        grad_desc(model, X, y, predictions, nn_hdim)

        #reset a, h, z
        model['h'] = []
        model['a'] = []
        model['z'] = []

    return model
    
def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
