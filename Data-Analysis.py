import MovieLens1MilDataHelper
import BookCrossingDataHelper
import JesterDataHelper
import numpy as np


class DataAnalysis(object):
    def __init__(self, train, test, data):
        self.train_matrix, self.test_matrix, self.data = train, test, data

    def sparsity(self):
        return 100 - ((self.get_total_ratings() / float(self.get_matrix_size())) * 100)

    def density(self):
        return 100 - self.sparsity()

    def get_users(self):
        return len(self.train_matrix)

    def get_items(self):
        return len(self.train_matrix[0])

    def get_total_ratings(self):
        train, test = self.train_matrix, self.test_matrix
        total_rated = 0
        for row in train:
            total_rated += np.count_nonzero(row)
        for row in test:
            total_rated += np.count_nonzero(row)
        return total_rated

    def get_matrix_size(self):
        return len(self.train_matrix) * len(self.train_matrix[0])


def to_string(arr):
    return_string = ""
    for string in arr:
        return_string += string
    return return_string


# String builder replication
StringBuilder = []

# The movielens dataset
print("MovieLens")
movie = MovieLens1MilDataHelper.MatrixCreation()
train_matrix, test_matrix = movie.get_train_test_matrix()
data_frame = movie.get_data_frame()
movieAnalysis = DataAnalysis(train_matrix, test_matrix, data_frame)
StringBuilder.append("MovieLens" + "\n")
StringBuilder.append("\tThe number of users {0}".format(movieAnalysis.get_users()) + "\n")
StringBuilder.append("\tThe number of items {0}".format(movieAnalysis.get_items()) + "\n")
StringBuilder.append("\tThe number of ratings {0}".format(movieAnalysis.get_total_ratings()) + "\n")
StringBuilder.append("\tThe sparsity level {0}".format(movieAnalysis.sparsity()) + "\n")
StringBuilder.append("\tThe density level {0}".format(movieAnalysis.density()) + "\n")

# The book crossing dataset
print("Book-Crossing")
book = BookCrossingDataHelper.MatrixCreation()
train_matrix, test_matrix = book.get_train_test_matrix()
data_frame = book.get_data_frame()
bookAnalysis = DataAnalysis(train_matrix, test_matrix, data_frame)
StringBuilder.append("Book-Crossing" + "\n")
StringBuilder.append("\tThe number of users {0}".format(bookAnalysis.get_users()) + "\n")
StringBuilder.append("\tThe number of items {0}".format(bookAnalysis.get_items()) + "\n")
StringBuilder.append("\tThe number of ratings {0}".format(bookAnalysis.get_total_ratings()) + "\n")
StringBuilder.append("\tThe sparsity level {0}".format(bookAnalysis.sparsity()) + "\n")
StringBuilder.append("\tThe density level {0}".format(bookAnalysis.density()) + "\n")

# The jester dataset
print("Jester")
jester = JesterDataHelper.MatrixCreation()
train_matrix, test_matrix = jester.get_train_test_matrix()
data_frame = jester.get_data_frame()
jesterAnalysis = DataAnalysis(train_matrix, test_matrix, data_frame)
StringBuilder.append("Jester" + "\n")
StringBuilder.append("\tThe number of users {0}".format(jesterAnalysis.get_users()) + "\n")
StringBuilder.append("\tThe number of items {0}".format(jesterAnalysis.get_items()) + "\n")
StringBuilder.append("\tThe number of ratings {0}".format(jesterAnalysis.get_total_ratings()) + "\n")
StringBuilder.append("\tThe sparsity level {0}".format(jesterAnalysis.sparsity()) + "\n")
StringBuilder.append("\tThe density level {0}".format(jesterAnalysis.density()) + "\n")

# Print the results to DataAnalysis
f1 = open('./information/DataAnalysis.txt', 'w+')
f1.write(to_string(StringBuilder))
