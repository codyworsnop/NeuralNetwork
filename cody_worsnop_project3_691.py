import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_moons


def calculate_loss(model, y, y_hat):

    #L(y, y_hat) = - 1/N * sum(y_n * log(y_hat_n))
    lossSum = 0.0

    for n in range(y.shape[0]):
        lossSum += np.log(y_hat[n][y[n]])

    return ((-1/y.shape[0]) * lossSum)

def predict(model, x, classification=False):

    #a = x*w1 + b1 
    a = np.dot(x, model['w1']) + model['b1']

    #h = tanh(a)
    h = np.tanh(a)

    #z = h*w2 + b2 
    z = np.dot(h, model['w2']) + model['b2']

    #yhat = softmax(z)
    softmax = np.exp(z) / np.sum(np.exp(z))

    #update model
    model['a'].append(a)
    model['h'].append(h)
    model['z'].append(z)


    if (classification):
        return np.argmax(softmax, axis=1)

    return softmax

def predict_many(model, X, outputSize=2):

    index = 0
    predictions = np.zeros((X.shape[0], outputSize))

    for sample in X:
        sample = np.reshape(sample, (1, 2))
        predictions[index] = predict(model, sample)
        index = index + 1

    return predictions

def grad_desc(model, X, y, y_hat, eta, nn_hdim):

    h = np.asarray([item for sublist in model['h'] for item in sublist])
    a = np.asarray([item for sublist in model['a'] for item in sublist])
    eta = 0.01
    diff_y = np.array(y_hat)

    for iteration in range(0, X.shape[0]):

        h_value = np.reshape(h[iteration], (1, nn_hdim))
        x_value = np.reshape(X[iteration], (1, 2))   

        #derivative of L with respect to y
        #y_hat - y 
        diff_y[iteration][y[iteration]] -= 1 
        dldy = np.reshape(diff_y[iteration], (1, 2)) 

        #derivative of L with respect to a
        #1 - tanh(a)^2 * dldy * w2^t
        tan_func = np.reshape((1 - (np.tanh(a[iteration]) ** 2)), (1, nn_hdim))
        dlda = tan_func * np.dot(dldy, np.transpose(model['w2']))

        #derivative of L with respect to w2
        #h_transpose * (y_hat - y)
        dlw2 = np.dot(np.transpose(h_value), dldy)

        #derivative of L with respect to b2 
        #y_hat - y
        dlb2 = dldy
        
        #derivative of L with respect to w1
        #x_transpose * (1 - tanh^2(a) * (y_hat - y) * w2_transpose)
        dlw1 = np.dot(np.transpose(x_value), dlda)

        #derivative of L with respect to b1
        #(1 - tanh^2(a) * (y_hat - y) * w2_transpose)
        dlb1 = dlda

        model['w1'] = model['w1'] - eta * dlw1
        model['b1'] = model['b1'] - eta * dlb1
        model['w2'] = model['w2'] - eta * dlw2
        model['b2'] = model['b2'] - eta * dlb2




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
        if (print_loss and iteration % 100 == 0 and iteration != 0):

            #calculate the loss
            loss = calculate_loss(model, y, predictions)

            print("Current loss value: " + str(loss))

        #back propagate to update the weights 
        grad_desc(model, X, y, predictions, 0.1, nn_hdim)

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
            loss = calculate_loss(model, y, predictions)

            print("Current loss value: " + str(loss))

        #back propagate to update the weights 
        grad_desc(model, X, y, predictions, 0.1, nn_hdim)

        #reset a, h, z
        model['h'] = []
        model['a'] = []
        model['z'] = []

    return model
    
def plot_decision_boundary(pred_func, X, y):

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.1

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

#X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)
X, y = make_moons(200, noise=0.2)
plt.figure(figsize=(16,32))
#hidden_layer_dimensions = [1, 2, 3, 4]
hidden_layer_dimensions = [4]


for i, nn_hdim in enumerate(hidden_layer_dimensions):
    print ("NEXT")
    plt.subplot(2, 2, i+1)
    plt.title('Hidden Layer Size %d' % nn_hdim)
    model = build_model(X, y, nn_hdim, 500, True)
    plot_decision_boundary(lambda x: predict(model, x, True), X, y)

plt.show()
a = 3
