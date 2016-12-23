# python file imports
import MovieLens100kDataHelper
import MovieLens1MilDataHelper
import BookCrossingDataHelper
import JesterDataHelper
# imports
import pandas as pd
import numpy as np
from math import sqrt
from random import randint
import numpy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv
import tensorflow as tf
import time

"""
Command line interface
- needs to be added
"""
train_matrix_data_helper, test_matrix_data_helper = None, None
item_max_count = None
# command line read the input in to select the data
data = int(input("Enter file\n1 for MovieLens\n2 for Book-Crossing\n3 for Jester\n"))
if data == 1:
    print("MovieLens")
    # movieLensIndicator = int(input("Enter\n1 for MovieLens100k\n2 for MovieLens1Millon\n"))
    movieLensIndicator = 2
    if movieLensIndicator == 1:
        print("MovieLens 100k")
        data_helper = MovieLens100kDataHelper.DataHelper()
        train_matrix_data_helper = data_helper.get_train_matrix()
        test_matrix_data_helper = data_helper.get_test_matrix()
        item_max_count = data_helper.get_item_max()
    elif movieLensIndicator == 2:
        print("MovieLens 1Million")
        data_helper = MovieLens1MilDataHelper.MatrixCreation()
        train_matrix_data_helper = data_helper.get_training_matrix()
        test_matrix_data_helper = data_helper.get_testing_matrix()
        item_max_count = data_helper.get_item_max()
elif data == 2:
    print("Book-Crossing")
    data_helper = BookCrossingDataHelper.MatrixCreation()
    train_matrix_data_helper = data_helper.get_train_matrix()
    test_matrix_data_helper = data_helper.get_test_matrix()
    item_max_count = data_helper.get_item_max()
elif data == 3:
    print("Jester")
    data_helper = JesterDataHelper.MatrixCreation()
    train_matrix_data_helper, test_matrix_data_helper = data_helper.get_train_test_matrix()
    item_max_count = data_helper.get_item_max()
print("Loaded the train and test matrices")

"""
Model Parameters
"""
t0 = time.time()
# Learning rate
alpha = .03
hidden_units = 3
epochs = 10
batch_size = 10
visible_units = item_max_count
train_matrix, test_matrix = train_matrix_data_helper, test_matrix_data_helper
print("Model parameters are set to default values")
print("Learning rate is set to {0}".format(alpha))
print("The number of hidden units is set to {0}".format(hidden_units))
print("The number of visible units is set to the number items in the train matrix {0}".format(visible_units))
print("The number of epochs is {0}".format(epochs))
print("The batchsize {0}".format(batch_size))

"""
Helper functions
"""


def get_batches(len_matrix, batch_size):
    batches = zip(range(0, len_matrix, batch_size), range(batch_size, len_matrix, batch_size))
    return batches


def set_non_zero(matrix):
    non_zero = matrix > 0
    non_zero[non_zero == True] = 1
    non_zero[non_zero == False] = 0
    return non_zero


def update_current_weight():
    return sess.run(update_weights,
                    feed_dict={visible_units_input: batch,
                               weight_placeholder: previous_weight,
                               visible_bias: previous_visible_bias,
                               hidden_bias: previous_hidden_bias})


def update_current_visable_bias():
    return sess.run(update_visible_bias,
                    feed_dict={visible_units_input: batch,
                               weight_placeholder: previous_weight,
                               visible_bias: previous_visible_bias,
                               hidden_bias: previous_hidden_bias})


def update_current_hidden_bias():
    return sess.run(update_hidden_bias,
                    feed_dict={visible_units_input: batch,
                               weight_placeholder: previous_weight,
                               visible_bias: previous_visible_bias,
                               hidden_bias: previous_hidden_bias})


"""
Setting the initial weights
- the weights are normally initialized to random values chosen from a zero mean Gaussian.
    - larger values can lead to a faster model, but may cause worse final model
    - Note should make sure the initial values do not drive the hidden unit probabilities close to 0 or 1.
- If the statistics used for learning are all stochastic the initial values can be set to zero
because the noise in the statistics will make the hidden units become different from one another
even if they all have identical connectives.
"""
# setting the placeholders for the visible, hidden, and weight
visible_bias = tf.placeholder("float", [visible_units])
hidden_bias = tf.placeholder("float", [hidden_units])
weight_placeholder = tf.placeholder("float", [visible_units, hidden_units])
# Input processing
visible_units_input = tf.placeholder("float", [None, visible_units])
"""
- Obtain the probabilities for the hidden units
    - For a single unit, this is computed by taking the sigmoid function of its total input.
- Hidden units turn on if the probability is greater than a random number uniformly
distributed between 0 and 1
- Note when using Contrastive Divergence only the final update of the hidden units should use probability
- Hidden units use binary values, otherwise each hidden unit can communicate a real value to
the visible units in the reconstruction phase
"""
hidden_units_probabilities = tf.nn.sigmoid(tf.matmul(visible_units_input, weight_placeholder) + hidden_bias)
hidden_unit_activation = tf.nn.relu(
    tf.sign(hidden_units_probabilities - tf.random_uniform(tf.shape(hidden_units_probabilities))))
# Positive Gradient
"""
- Computing the gradients (wikipedia)
- Positive gradient compute the outer product of the visible_units_input and the hidden_units_activation
"""
w_pos_grad = tf.matmul(tf.transpose(visible_units_input), hidden_unit_activation)
# Reconstruction
"""
- Common to use the probability, instead of sampling a random binary value
- Probability method for hidden units?
"""
visible_units_probabilities_reconstruction = tf.nn.softmax(
    tf.matmul(hidden_unit_activation, tf.transpose(weight_placeholder)) + visible_bias)
visible_units_activation = tf.nn.relu(tf.sign(visible_units_probabilities_reconstruction - tf.random_uniform(
    tf.shape(visible_units_probabilities_reconstruction))))
"""
- Note when using Contrastive Divergence only the final update of the hidden units should use probability
- Final update of the hidden units using probabilities
"""
hidden_units_final_probabilities = tf.nn.sigmoid(
    tf.matmul(visible_units_activation, weight_placeholder) + hidden_bias)
# Negative Gradient
"""
- Computing the gradients (wikipedia)
- Negative gradient compute the outer product of the visible_units_activation and the hidden_units_final_probabilities
"""
w_neg_grad = tf.matmul(tf.transpose(visible_units_activation), hidden_units_final_probabilities)
"""
- Contrastive Divergence take the difference between the positive and negative gradients
"""
# Calculate the Contrastive Divergence to maximize
CD = (w_pos_grad - w_neg_grad)
"""
Updating the weights and the biases
- from wikipedia?
- take the difference between the initial and the activation vectors of the visible and hidden units
"""
update_weights = weight_placeholder + alpha * CD
update_visible_bias = visible_bias + alpha * tf.reduce_mean(visible_units_input - visible_units_activation, 0)
update_hidden_bias = hidden_bias + alpha * tf.reduce_mean(
    hidden_unit_activation - hidden_units_final_probabilities, 0)
"""
Set the vectors for the current and previous weights.
- weights is a matrix
"""
current_weight = np.zeros([visible_units, hidden_units], np.float32)
current_visible_bias = np.zeros([visible_units], np.float32)
current_hidden_bias = np.zeros([hidden_units], np.float32)
previous_weight = np.zeros([visible_units, hidden_units], np.float32)
previous_visible_bias = np.zeros([visible_units], np.float32)
previous_hidden_bias = np.zeros([hidden_units], np.float32)
"""
Start the tensor flow session and initialize all the variables
"""
sess = tf.Session()
sess.run(tf.initialize_all_variables())
"""
Build the model
"""
errors = []
print("Building model")
for i in range(epochs):
    batch_array = get_batches(len(train_matrix) + batch_size, batch_size)
    for start, end in batch_array:
        batch = train_matrix[start:end]
        current_weight = update_current_weight()
        current_visible_bias = update_current_visable_bias()
        current_hidden_bias = update_current_hidden_bias()
        previous_weight = current_weight
        previous_visible_bias = current_visible_bias
        previous_hidden_bias = current_hidden_bias
    print("\tEpoch ({0},{1})".format(i + 1, epochs))
print("Model built")

"""
Obtain the RMSE error of test_matrix
- needs to be added
"""
print("\nRMSE")
print("Obtaining the prediction matrix")
prediction_matrix = np.zeros((train_matrix.shape[0], train_matrix.shape[1]))
print(len(train_matrix))
counter = 0
for i, x in enumerate(train_matrix):
    if counter == 100:
        print(i)
        counter = 0
    counter += 1
    user_row = [x]
    hidden_prediction = tf.nn.sigmoid(tf.matmul(visible_units_input, weight_placeholder) + hidden_bias)
    visible_prediction = tf.nn.sigmoid(tf.matmul(hidden_prediction, tf.transpose(weight_placeholder)) + visible_bias)
    iteration_probabilities = sess.run(hidden_prediction, feed_dict={visible_units_input: user_row, weight_placeholder: previous_weight,
                                                                     hidden_bias: previous_hidden_bias})
    iteration_probabilities2 = sess.run(visible_prediction, feed_dict={hidden_prediction: iteration_probabilities, weight_placeholder: previous_weight, visible_bias: previous_visible_bias})
    user_ratings = iteration_probabilities2[0]
    prediction_matrix[i] = user_ratings
print("Calculating the RMSE")

train_matrix, non_zero_train = train_matrix, set_non_zero(train_matrix)
rmse_train = np.sqrt(np.sum((non_zero_train * (train_matrix - prediction_matrix)) ** 2) / np.sum(non_zero_train))
print("The RMSE-train {0}".format(rmse_train))

test_matrix, non_zero_test = test_matrix, set_non_zero(test_matrix)
rmse_test = np.sqrt(np.sum((non_zero_test * (test_matrix - prediction_matrix)) ** 2) / np.sum(non_zero_test))
print("The RMSE-test {0}".format(rmse_test))
t4 = time.time()
total = (t4 - t0) / float(60)
x = ("\nrun-time {0}".format(round(total, 6)))
print(x)
