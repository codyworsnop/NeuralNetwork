import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def calculate_loss(model, X, y, y_hat):

    #init values
    index = 0
    incorrectPredictions = 0

    #find the prediction for every sample, calculating the error along the way 
    for sample in X:

        #get prediction and save it to y_hat for back propagation
        prediction = predict(model, sample)
        y_hat[index] = prediction

        #get the label
        label = y[index]
        
        #check if our prediction is correct
        if (label * prediction <= 0):
            incorrectPredictions += 1

        #increment the index value
        index += 1

    return incorrectPredictions / X.shape[0]

def predict(model, x):
    
    #a = x*w1 + b1 
    a = np.dot(x, model['w1']) + model['b1']

    #h = tanh(a)
    h = np.tanh(a)

    #z = h*w2 + b2 
    z = np.dot(h, model['w2']) + model['b2']

    #yhat = softmax(z)
    softmax = np.exp(z) / np.sum(np.exp(z))

    #update model jj
    model['a'] = a
    model['h'] = h
    model['z'] = z
    
    return softmax


def back_prop(model, y, y_hat, eta):

    for index in range(y.shape[0]):

        #derivative of L with respect to w1
        #x_transpose * (1 - tanh^2(a) * (y_hat - y) * w2_transpose)
        model['w1'] = model['w1'] - eta * ((1 - np.tanh(model['a'])) ** 2) * (y_hat[index] - y[index]) * np.transpose(model['w2'])

        #derivative of L with respect to w2
        #h_transpose * (y_hat - y)
        model['w2'] = model['w2'] - eta * (np.transpose(model['h']) * (y_hat[index] - y[index]))

        #derivative of L with respect to b1
        #(1 - tanh^2(a) * (y_hat - y) * w2_transpose)
        model['b1'] = model['b1'] - eta * ((1 - np.tanh(model['a'])) ** 2) * (y_hat[index] - y[index]) * np.transpose(model['w2'])

        #derivative of L with respoect to b2
        #(y_hat - y)
        model['b2'] = model['b2'] - eta * (y_hat[index] - y[index]) 
    


def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    
    #initialize our prediction matrix
    y_hat = np.zeros(y.shape, dtype=np.int)

    #initialize weights randomly over the normal distribution, let bias terms init to zero
    model = {
    'w1' : np.random.randn(2, nn_hdim),
    'b1' : 0,
    'w2' : np.random.randn(nn_hdim, nn_hdim),
    'b2' : 0,
    'h' : 0,
    'a' : 0,
    'z' : 0,  
    }
    
    for iteration in range(num_passes):

        #forward propagate and calculate the loss
        loss = calculate_loss(model, X, y, y_hat)

        #print loss if needed
        if (print_loss and iteration % 1000 == 0):

            print("Current loss value: " + str(loss))

        #back propagate to update the weights 
        back_prop(model, y, y_hat, 0.1)

        print ("Iteration: " + str(iteration))

    return model
    
    
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    h = 0.01 

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, 1)#, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y)#, cmap=plt.cm.Spectral)


X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)

plt.figure(figsize=(16,32))
hidden_layer_dimensions = [1, 2, 3, 4]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(X, y, nn_hdim, 500)
    plot_decision_boundary(lambda x: predict(model, x))


plt.show()
