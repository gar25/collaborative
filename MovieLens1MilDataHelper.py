# imports
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv


class DataHelper(object):
    def __init__(self):
        self.movies_df = pd.read_csv('data/movies.dat', sep='::', names=['MovieID', 'Title', 'Genres'], engine="python")
        self.ratings_df = pd.read_csv('data/ratings.dat', sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                                      engine="python")

    def get_main_data(self):
        return self.data_munging()

    def data_munging(self):
        self.movies_df['Index'] = self.movies_df.index
        merge_df = self.movies_df.merge(self.ratings_df, on='MovieID')
        return merge_df


class MatrixCreation(DataHelper):
    def __init__(self):
        DataHelper.__init__(self)
        self.main_data_frame = self.get_main_data()
        self.user_max, self.item_max = self.set_user_max(), self.set_item_max()
        self.train_matrix, self.test_matrix = np.zeros((self.user_max, self.item_max)), np.zeros(
            (self.user_max, self.item_max))
        self.create_train_test()

    def set_item_max(self):
        return self.main_data_frame['MovieID'].max()

    def set_user_max(self):
        return self.main_data_frame['UserID'].max()

    def create_train_test(self):
        train_data, test_data = cv.train_test_split(self.main_data_frame, test_size=0.20)
        train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
        train_matrix, test_matrix = self.train_matrix, self.test_matrix
        for row in train_data.itertuples():
            userIndex, movieIndex, rating = row.UserID, row.MovieID, row.Rating
            train_matrix[userIndex - 1][movieIndex - 1] = rating
        for row in test_data.itertuples():
            userIndex, movieIndex, rating = row.UserID, row.MovieID, row.Rating
            test_matrix[userIndex - 1][movieIndex - 1] = rating
        self.train_matrix, self.test_matrix = train_matrix, test_matrix

    def get_training_matrix(self):
        return self.train_matrix

    def get_testing_matrix(self):
        return self.test_matrix

    def get_train_test_matrix(self):
        return self.train_matrix, self.test_matrix

    def get_data_frame(self):
        return self.main_data_frame

    def get_user_max(self):
        return self.main_data_frame['UserID'].max()

    def get_item_max(self):
        return self.main_data_frame['MovieID'].max()
