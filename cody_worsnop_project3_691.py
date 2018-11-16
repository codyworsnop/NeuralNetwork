import numpy as np
import sklearn
import matplotlib.pyplot as plot
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=0)

plot.figure(figsize=(16,32))
hidden_layer_dimensions = [1, 2, 3, 4]

def calculate_loss(model, X, y):
    print("Not implemented")

def predict(model, x):
    print("Not implemented")

def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
    print("Not implemented")
    
def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, x[:, 1].max() + 0.5

    h = 0.01 

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plot.subplot(5, 2, i+1)
    plot.title('Hidden Layer Size %d' % nn_hdim)
    #build_model(X, y, nn_hdim)
    #plot_decision_boundary(lambda x: predict(model, x))

plot.show()