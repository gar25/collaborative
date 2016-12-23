import pandas as pd
import numpy as np
from sklearn import cross_validation as cv


class DataHelper(object):
    def __init__(self):
        self.data_frame = pd.read_csv('data/jester.csv', sep=';')


class MatrixCreation(DataHelper):
    train = []
    test = []

    def __init__(self):
        DataHelper.__init__(self)

    def get_train_test_matrix(self):
        data_frame = self.data_frame
        data_frame[data_frame == 99] = -11
        number_users, number_items = len(data_frame), len(data_frame.iloc[0].tolist()[0].split(",")[1:])
        number_users = 3500
        train_data, test_data = cv.train_test_split(data_frame, test_size=0.20)
        train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
        train_matrix, test_matrix = np.zeros((number_users, number_items)), np.zeros((number_users, number_items))
        user_index = np.random.permutation(number_users).tolist()
        train_matrix_counter, test_matrix_counter = int(number_users * .8), int(number_users * .2)
        for x, row in train_data.iterrows():
            train_matrix_counter -= 1
            row = row.tolist()[0].split(",")
            row = np.array(row[1:], dtype='f')
            for i, value in enumerate(row):
                row[i] = 0 if value == 99 else value + 11
            train_matrix[user_index.pop()] = row
            if train_matrix_counter == 0:
                break
        for x, row in test_data.iterrows():
            test_matrix_counter -= 1
            row = row.tolist()[0].split(",")
            row = np.array(row[1:], dtype='f')
            for i, value in enumerate(row):
                row[i] = 0 if value == 99 else value + 11
            test_matrix[user_index.pop()] = row
            if test_matrix_counter == 0:
                break
        return train_matrix, test_matrix

    def get_data_frame(self):
        return self.data_frame

    def get_user_max(self):
        return 3500

    def get_item_max(self):
        return 100
