import pandas as pd
import numpy as np

import MovieLens1MilDataHelper

"""
matrixCreationObject = MovieLensDataHelper.MatrixCreation()
train_matrix = matrixCreationObject.get_training_matrix()
test_matrix = matrixCreationObject.get_testing_matrix()
"""

import BookCrossingDataHelper
from sklearn import cross_validation as cv

"""
data_helper = BookCrossingDataHelper.MatrixCreation()
"""

import JesterDataHelper

""""
data_helper = JesterDataHelper.MatrixCreation()
train, test = data_helper.get_train_test_matrix()
print(train.shape)
print(test.shape)
print(len(train == test))

"""

