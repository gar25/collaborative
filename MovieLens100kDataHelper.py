import numpy as np
import pandas as pd
from sklearn import cross_validation as cv


class DataHelper(object):
    def __init__(self):
        self.data_frame = self.set_data_frame()
        self.max_user, self.max_item = self.set_max_user, self.set_max_item
        self.train_matrix, self.test_matrix = self.get_train_test_matrix()

    @staticmethod
    def set_data_frame():
        df = pd.read_csv('data/100ku.data', sep='\t', usecols=pd.read_csv('100ku.data', sep='\t', nrows=1).columns[:3])
        df.columns = ['user', 'item', 'rating']
        return df

    def set_max_user(self):
        self.max_user = self.data_frame['user'].max()

    def set_max_item(self):
        self.max_item = self.data_frame['item'].max()

    def get_train_test_matrix(self):
        train_data, test_data = cv.train_test_split(self.data_frame, test_size=0.2)
        train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
        train_matrix = test_matrix = np.zeros((self.max_user, self.max_item))
        for row in train_data.itertuples():
            user_index, item_index = row.user - 1, row.item - 1
            train_matrix[user_index][item_index] = row.rating
        for row in test_data.itertuples():
            user_index, item_index = row.user - 1, row.item - 1
            test_matrix[user_index][item_index] = row.rating
        return train_matrix, test_matrix

    def get_train_matrix(self):
        return self.train_matrix

    def get_test_matrix(self):
        return self.test_matrix

    def get_item_max(self):
        return self.max_item

    def get_user_max(self):
        return self.max_user
