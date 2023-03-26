import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common_v3 import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X,Y = load_coffee_data()
print(X.shape, Y.shape)

plt_roast(X,Y)

# now normalizing the data
# where features in the data are each normalized to have a similar range.
# The procedure below uses a Keras normalization layer. It has the following steps:
# create a "Normalization Layer". Note, as applied here, this is not a layer in your model.
# 'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
# normalize the data.
# It is important to apply normalization to any future data that utilizes the learned model.

print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# Tile/copy our data to increase the training set size and reduce the number of training epochs.
# The np.tile() function in NumPy is used to create a new array by repeating an existing array a certain number of times
# along one or more dimensions.
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))
print(Xt.shape, Yt.shape)

# By creating a larger training set, it can sometimes be possible to reduce the number of training epochs needed to achieve a satisfactory level
# of accuracy or convergence. This is because with a larger dataset,
# the model has more data to learn from and can potentially generalize better to new, unseen data.


## now let's build the Coffee Roasting Network" described in lecture. There are two layers with sigmoid activations

tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)

model.summary()

W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)

# let's exam the weights of each layer
# layer 1 has 3 neurons,
# each neuron has a W vector and b so it would be W1, W2 corresponding featurea X1, X2 (duration and temperature)
# layer 2 has 1 neuron

# model.compile statement defines a loss function and specifies a compile optimization.
# model.fit statement runs gradient descent and fits the weights to the data
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,
    epochs=10,
)

# after fitting, the weights have been updated:
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)


# now to set a weights to certain number for our exercise later
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])

X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

# now let's understand how each node in the layer works
# We will plot the output of each node for all values of the inputs (duration,temp). Each unit is a logistic function whose output can range from zero to one.
# The shading in the graph represents the output value.
plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)

# The shading shows that each unit is responsible for a different "bad roast" region.
# unit 0 has larger values when the temperature is too low.
# unit 1 has larger values when the duration is too short
# and unit 2 has larger values for bad combinations of time/temp.
# It is worth noting that the network learned these functions on its own through the process of gradient descent.
# They are very much the same sort of functions a person might choose to make the same decisions.
#
# The function plot of the final layer is a bit more difficult to visualize.
# It's inputs are the output of the first layer. We know that the first layer uses sigmoids so their output range is between zero and one. We can create a 3-D plot that calculates the output for all possible combinations of the three inputs. This is shown below. Above, high output values correspond to 'bad roast' area's. Below, the maximum output is in area's where the three inputs are small values corresponding to 'good roast' area's.
plt_output_unit(W2,b2)

# The final graph shows the whole network in action.
# The left graph is the raw output of the final layer represented by the blue shading. This is overlaid on the training data represented by the X's and O's.
# The right graph is the output of the network after a decision threshold. The X's and O's here correspond to decisions made by the network.
# The following takes a moment to run

netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)
plt.show()