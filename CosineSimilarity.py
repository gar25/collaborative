import MovieLens1MilDataHelper
import BookCrossingDataHelper
import JesterDataHelper
import MovieLens100kDataHelper
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
import time
from pathlib import Path


class CosineSimilarity(object):
    def __init__(self, train_matrix, test_matrix):
        self.train_matrix, self.test_matrix = train_matrix, test_matrix
        self.user_similarity = self.get_user_prediction()
        self.prediction_user = self.get_predictions_user(self.train_matrix, self.user_similarity)

    def get_user_prediction(self):
        return pairwise_distances(self.train_matrix, metric='cosine')

    @staticmethod
    def get_predictions_user(rating_matrix, similarity):
        differenceRating = rating_matrix - rating_matrix.mean(axis=1)[:, np.newaxis]
        numerator = rating_matrix.mean(axis=1)[:, np.newaxis] + similarity.dot(differenceRating)
        denominator = np.array([np.abs(similarity).sum(axis=1)]).T
        predictionMatrix = (numerator / denominator)
        return predictionMatrix

    @staticmethod
    def get_rmse(prediction_matrix, labels):
        label_indices = labels.nonzero()
        prediction_matrix = prediction_matrix[label_indices].flatten()
        labels = labels[label_indices].flatten()
        return sqrt(mean_squared_error(prediction_matrix, labels))


def to_string(arr):
    return_string = ""
    for string in arr:
        return_string += string
    return return_string


def main():
    # String builder replication
    StringBuilder = []
    t0 = time.time()
    print("Cosine Similarity")
    data = int(input("Enter file\n1 for MovieLens\n2 for Book-Crossing\n3 for Jester\n4 for all\n"))
    if data == 1 or data == 4 or data == 5:
        print("MovieLens")
        movieLensIndicator = int(input("Enter\n1 for MovieLens100k\n2 for MovieLens1Millon\n"))
        if movieLensIndicator == 2:
            my_file = Path("numpy_data/MovieLensTraining.npy")
            if my_file.is_file():
                print("\tLoading files")
                train_matrix = np.load('numpy_data/MovieLensTraining.npy')
                test_matrix = np.load('numpy_data/MovieLensTesting.npy')
                print("\tFiles loaded")
            else:
                print("Loading Matrix creation")
                matrixCreationObject = MovieLens1MilDataHelper.MatrixCreation()
                train_matrix = matrixCreationObject.get_training_matrix()
                test_matrix = matrixCreationObject.get_testing_matrix()
                np.save('numpy_data/MovieLensTraining.npy', train_matrix)
                np.save('numpy_data/MovieLensTesting.npy', test_matrix)
        else:
            my_file = Path("numpy_data/MovieLensTrain100k.npy")
            if my_file.is_file():
                print("\tLoading files")
                train_matrix = np.load('numpy_data/MovieLensTrain100k.npy')
                test_matrix = np.load('numpy_data/MovieLensTest100k.npy')
                print("\tFiles loaded")
            else:
                print("Loading Matrix creation")
                matrixCreationObject = MovieLens100kDataHelper.DataHelper()
                train_matrix = matrixCreationObject.get_train_matrix()
                test_matrix = matrixCreationObject.get_test_matrix()
                np.save('numpy_data/MovieLensTrain100k.npy', train_matrix)
                np.save('numpy_data/MovieLensTest100k.npy', test_matrix)
        time_start = time.time()
        cosine_object = CosineSimilarity(train_matrix, test_matrix)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("MovieLens\n\t")
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.train_matrix)
        StringBuilder.append("{0}\n\t".format(rmse))
        print('\tCosineSimilarity\n\t\t Train-RMSE {0}'.format(round(rmse, 4)))
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.test_matrix)
        StringBuilder.append('{0}\n'.format(rmse))
        print('\tCosineSimilarity\n\t\t Test-RMSE {0}'.format(round(rmse, 4)))
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    if data == 2 or data == 4 or data == 5:
        print("Book-Crossing")
        my_file = Path("numpy_data/Book-CrossingTraining.npy")
        if my_file.is_file():
            print("\tLoading files")
            train_matrix = np.load('numpy_data/Book-CrossingTraining.npy')
            test_matrix = np.load('numpy_data/Book-CrossingTesting.npy')
            print("\tFiles loaded")
        else:
            print("Loading Matrix creation")
            matrixCreationObject = BookCrossingDataHelper.MatrixCreation()
            train_matrix = matrixCreationObject.get_train_matrix()
            test_matrix = matrixCreationObject.get_test_matrix()
            np.save('numpy_data/Book-CrossingTraining.npy', train_matrix)
            np.save('numpy_data/Book-CrossingTesting.npy', test_matrix)
        cosine_object = CosineSimilarity(train_matrix, test_matrix)
        print("\n\t---------------------------------------------------------------")
        time_start = time.time()
        StringBuilder.append("Book-Crossing\n\t")
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.train_matrix)
        StringBuilder.append("{0}\n\t".format(rmse))
        print('\tCosineSimilarity\n\t\t Train-RMSE {0}'.format(round(rmse, 4)))
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.test_matrix)
        StringBuilder.append("{0}\n".format(rmse))
        print('\tCosineSimilarity\n\t\t Test-RMSE {0}'.format(round(rmse, 4)))
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    if data == 3 or data == 4 or data == 5:
        print("Jester")
        my_file = Path("numpy_data/Jester-Training.npy")
        if my_file.is_file():
            print("\tLoading files")
            train_matrix = np.load('numpy_data/Jester-Training.npy')
            test_matrix = np.load('numpy_data/Jester-Testing.npy')
            print("\tFiles loaded")
        else:
            print("Loading Matrix creation")
            matrixCreationObject = JesterDataHelper.MatrixCreation()
            train_matrix, test_matrix = matrixCreationObject.get_train_test_matrix()
            np.save('numpy_data/Jester-Training.npy', train_matrix)
            np.save('numpy_data/Jester-Testing.npy', test_matrix)
        time_start = time.time()
        cosine_object = CosineSimilarity(train_matrix, test_matrix)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("Jester\n\t")
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.train_matrix)
        StringBuilder.append("{0}\n\t".format(rmse))
        print('\tCosineSimilarity\n\t\t Train-RMSE {0}'.format(round(rmse, 4)))
        rmse = cosine_object.get_rmse(cosine_object.prediction_user, cosine_object.test_matrix)
        StringBuilder.append("{0}\n".format(rmse))
        print('\tCosineSimilarity\n\t\t Test-RMSE {0}'.format(round(rmse, 4)))
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    print("\n\n")
    t4 = time.time()
    total = (t4 - t0) / float(60)
    print("\nrun time {0}".format(round(total, 6)))
    if data == 5:
        StringBuilder.append("\nTotal run-time {0}".format(total))
        f1 = open('./information/Cosine-Results.txt', 'w+')
        f1.write(to_string(StringBuilder))


if __name__ == "__main__":
    main()
