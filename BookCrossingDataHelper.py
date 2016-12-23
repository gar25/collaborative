import numpy as np
import pandas as pd
from sklearn import cross_validation as cv


class DataHelper(object):
    def __init__(self):
        self.ratings_data_frame = pd.read_csv('data/BX-Book-Ratings.csv', sep=';', encoding='ISO-8859-1')
        self.books_data_frame = pd.read_csv('data/BX-Books.csv', sep=';',
                                            usecols=pd.read_csv('data/BX-Books.csv', sep=';', nrows=1).columns[:1],
                                            error_bad_lines=False)
        self.data_munging()
        self.max_user, self.max_item = 0, 0
        self.main_data_frame = self.set_main_data_frame()
        self.reduce_data(4000)

    def data_munging(self):
        self.ratings_data_frame.columns = ['UserID', 'ISBN', 'Rating']
        self.books_data_frame['Index'] = self.books_data_frame.index

    def set_main_data_frame(self):
        return pd.merge(self.ratings_data_frame, self.books_data_frame, how='inner', on='ISBN')

    def read_data(self):
        self.max_user = self.main_data_frame['UserID'].max()
        self.max_item = len(self.books_data_frame.groupby('ISBN'))

    def reduce_data(self, reduced_book_size):
        self.books_data_frame = self.books_data_frame[self.books_data_frame.Index < reduced_book_size]
        self.main_data_frame = pd.merge(self.ratings_data_frame, self.books_data_frame, how='inner', on='ISBN')
        self.read_data()


class MatrixCreation(DataHelper):
    def __init__(self):
        DataHelper.__init__(self)
        self.train_matrix, self.test_matrix = self.set_matrix(), self.set_matrix()
        self.create_reduced_matrix(10000)

    def set_matrix(self):
        return np.zeros((self.max_user, self.max_item))

    def create_reduced_matrix(self, number_users):
        train_matrix_counter, test_matrix_counter = int(number_users * .8), int(number_users * .2)
        user_index = np.random.permutation(number_users).tolist()
        train_data, test_data = cv.train_test_split(self.main_data_frame, test_size=0.20)
        train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
        train_matrix, test_matrix = np.zeros((number_users, self.max_item)), np.zeros((number_users, self.max_item))
        user_train_matrix, user_test_matrix = train_data.groupby('UserID'), test_data.groupby('UserID')
        for x, user_group in user_train_matrix:
            train_matrix_counter -= 1
            index_user = user_index.pop()
            for x1, row in user_group.iterrows():
                item_index = row.Index
                train_matrix[index_user][item_index] = row.Rating
            if train_matrix_counter == 0:
                break
        for x, user_group in user_test_matrix:
            test_matrix_counter -= 1
            index_user = user_index.pop()
            for x1, row in user_group.iterrows():
                item_index = row.Index
                test_matrix[index_user][item_index] = row.Rating
            if test_matrix_counter == 0:
                break
        self.train_matrix, self.test_matrix = train_matrix, test_matrix

    def create_matrix(self):
        return None

    def get_train_test_matrix(self):
        return self.train_matrix, self.test_matrix

    def get_train_matrix(self):
        return self.train_matrix

    def get_test_matrix(self):
        return self.test_matrix

    def get_data_frame(self):
        return self.main_data_frame

    def get_item_max(self):
        return self.max_item

    def get_user_max(self):
        return self.max_user
