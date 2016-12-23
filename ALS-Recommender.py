import MovieLens1MilDataHelper
import MovieLens100kDataHelper
import BookCrossingDataHelper
import JesterDataHelper
import numpy as np
from pathlib import Path
import time


# Alternating Least Squares
class ALS(object):
    def __init__(self, train_matrix, test_matrix, k, lambda_val, data_range):
        self.train_matrix, self.test_matrix = train_matrix, test_matrix
        self.lambda_val, self.k, self.data_range, self.E = lambda_val, k, data_range, np.eye(k)
        self.m, self.n = train_matrix.shape
        self.item_matrix = np.random.rand(self.k, self.n) * self.data_range
        self.user_matrix = np.random.rand(self.m, self.k) * self.data_range
        self.non_zero_train, self.non_zero_test = self.get_non_zero()
        self.set_item_matrix(), self.set_user_matrix()

    def set_item_matrix(self):
        self.item_matrix[0, :] = self.train_matrix.mean(axis=0)

    def get_item_matrix(self):
        return self.item_matrix

    def get_non_zero(self):
        non_zero_train = self.train_matrix > 0
        non_zero_train[non_zero_train == True] = 1
        non_zero_train[non_zero_train == False] = 0
        non_zero_test = self.test_matrix > 0
        non_zero_test[non_zero_test == True] = 1
        non_zero_test[non_zero_test == False] = 0
        return non_zero_train, non_zero_test

    def set_user_matrix(self):
        pass

    def get_user_matrix(self):
        return self.user_matrix

    @staticmethod
    def prediction(p, q):
        return np.dot(p, q)

    def rmse_train(self, q, p):
        predictionMatrix = self.prediction(p, q)
        train_matrix, non_zero_train = self.train_matrix, self.non_zero_train
        return np.sqrt(np.sum((non_zero_train * (train_matrix - predictionMatrix)) ** 2) / np.sum(non_zero_train))

    def rmse_test(self, q, p):
        predictionMatrix = self.prediction(p, q)
        test_matrix, non_zero_test = self.test_matrix, self.non_zero_test
        return np.sqrt(np.sum((non_zero_test * (test_matrix - predictionMatrix)) ** 2) / np.sum(non_zero_test))

    def algorithm(self):
        prev_train_rmse, prev_testing_rmse = float("Inf"), float("Inf")
        user_matrix, item_matrix = self.get_user_matrix(), self.get_item_matrix()
        regularization, E = self.lambda_val, self.E
        train_matrix, test_matrix = self.train_matrix, self.test_matrix
        non_zero_train, non_zero_test = self.non_zero_train, self.non_zero_test
        epoch = 0
        print("\tALS")
        while True:
            for i, rowi in enumerate(non_zero_train):
                nui = np.count_nonzero(rowi)
                nui = nui if nui != 0 else 1
                Ai = np.dot(item_matrix, np.dot(np.diag(rowi), item_matrix.T)) + regularization * nui * E
                Vi = np.dot(item_matrix, np.dot(np.diag(rowi), train_matrix[i].T))
                user_matrix[i, :] = np.linalg.solve(Ai, Vi)
            train_rmse = self.rmse_train(item_matrix, user_matrix)
            test_rmse = self.rmse_test(item_matrix, user_matrix)
            print('\t\tSolved for low rank user matrix\n\t\tepoch {0}  Training-RMSE {1}  Testing-RMSE {2}'.format(
                round(epoch, 2), round(train_rmse, 5),
                round(test_rmse, 5)))
            for j, rowj in enumerate(non_zero_train.T):
                nmj = np.count_nonzero(rowj)
                nmj = nmj if nmj != 0 else 1
                Aj = np.dot(user_matrix.T, np.dot(np.diag(rowj), user_matrix)) + regularization * nmj * E
                Vj = np.dot(user_matrix.T, np.dot(np.diag(rowj), train_matrix[:, j]))
                item_matrix[:, j] = np.linalg.solve(Aj, Vj)
            train_rmse = self.rmse_train(item_matrix, user_matrix)
            test_rmse = self.rmse_test(item_matrix, user_matrix)
            print('\t\tSolved for low rank item matrix\n\t\tepoch {0}  Training-RMSE {1}  Testing-RMSE {2}'.format(
                round(epoch, 2), round(train_rmse, 5),
                round(test_rmse, 5)))
            if test_rmse >= prev_testing_rmse or epoch is 5:
                break
            prev_train_rmse = train_rmse
            prev_testing_rmse = test_rmse
            self.user_matrix = user_matrix
            self.item_matrix = item_matrix
            epoch += 1
        x = ('\tFinal Training-RMSE {0}\n\tFinal Testing-RMSE {1}\n'.format(prev_train_rmse, prev_testing_rmse))
        print(x)
        return x


def to_string(arr):
    return_string = ""
    for string in arr:
        return_string += string
    return return_string


def main():
    # String builder replication
    StringBuilder = []
    t0 = time.time()
    print("Alternating Least Squares")
    data = int(input("Enter file\n1 for MovieLens\n2 for Book-Crossing\n3 for Jester\n4 for all\n"))
    if data == 1 or data == 4 or data == 6:
        print("MovieLens")
        movieLensIndicator, lambdaVal, k, data_range = 0, 0, 0, 0
        if data != 5:
            movieLensIndicator = int(input("Enter\n1 for MovieLens100k\n2 for MovieLens1Millon\n"))
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaVal = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 5 if data_range_input == 0 else data_range_input
        if data == 5:
            movieLensIndicator = 2
            lambdaVal = .1
            k = 3
            data_range = 5
        if movieLensIndicator == 2:
            my_file = Path("numpy_data/MovieLensTraining.npy")
            if my_file.is_file():
                print("\tLoading files")
                trainingMatrix = np.load('numpy_data/MovieLensTraining.npy')
                testingMatrix = np.load('numpy_data/MovieLensTesting.npy')
                print("\tFiles loaded")
            else:
                print("Loading Matrix creation")
                matrixCreationObject = MovieLens1MilDataHelper.MatrixCreation()
                trainingMatrix = matrixCreationObject.get_training_matrix()
                testingMatrix = matrixCreationObject.get_testing_matrix()
                np.save('numpy_data/MovieLensTraining.npy', trainingMatrix)
                np.save('numpy_data/MovieLensTesting.npy', testingMatrix)
        else:
            my_file = Path("numpy_data/MovieLensTrain100k.npy")
            if my_file.is_file():
                print("\tLoading files")
                trainingMatrix = np.load('numpy_data/MovieLensTrain100k.npy')
                testingMatrix = np.load('numpy_data/MovieLensTest100k.npy')
                print("\tFiles loaded")
            else:
                print("Loading Matrix creation")
                matrixCreationObject = MovieLens100kDataHelper.DataHelper()
                trainingMatrix = matrixCreationObject.get_train_matrix()
                testingMatrix = matrixCreationObject.get_test_matrix()
                np.save('numpy_data/MovieLensTrain100k.npy', trainingMatrix)
                np.save('numpy_data/MovieLensTest100k.npy', testingMatrix)
        time_start = time.time()
        alsObject = ALS(trainingMatrix, testingMatrix, k, lambdaVal, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("MovieLens\n")
        StringBuilder.append(alsObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    if data == 2 or data == 4 or data == 6:
        print("Book-Crossing")
        lambdaVal, k, data_range = 0, 0, 0
        if data != 5:
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaVal = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 2 if data_range_input == 0 else data_range_input
        if data == 5:
            lambdaVal = .1
            k = 3
            data_range = 5
        my_file = Path("numpy_data/Book-CrossingTraining.npy")
        if my_file.is_file():
            print("\tLoading files")
            trainingMatrix = np.load('numpy_data/Book-CrossingTraining.npy')
            testingMatrix = np.load('numpy_data/Book-CrossingTesting.npy')
            print("\tFiles loaded")
        else:
            print("Loading Matrix creation")
            matrixCreationObject = BookCrossingDataHelper.MatrixCreation()
            trainingMatrix = matrixCreationObject.get_train_matrix()
            testingMatrix = matrixCreationObject.get_test_matrix()
            np.save('numpy_data/Book-CrossingTraining.npy', trainingMatrix)
            np.save('numpy_data/Book-CrossingTesting.npy', testingMatrix)
        time_start = time.time()
        alsObject = ALS(trainingMatrix, testingMatrix, k, lambdaVal, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("Book-Crossing\n")
        StringBuilder.append(alsObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    if data == 3 or data == 4 or data == 5:
        print("Jester")
        lambdaVal, k, data_range = 0, 0, 0
        if data != 5:
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaVal = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 2 if data_range_input == 0 else data_range_input
        if data == 5:
            lambdaVal = .1
            k = 3
            data_range = 5
        my_file = Path("numpy_data/Jester-Training.npy")
        if my_file.is_file():
            print("\tLoading files")
            trainingMatrix = np.load('numpy_data/Jester-Training.npy')
            testingMatrix = np.load('numpy_data/Jester-Testing.npy')
            print("\tFiles loaded")
        else:
            print("Loading Matrix creation")
            matrixCreationObject = JesterDataHelper.MatrixCreation()
            trainingMatrix, testingMatrix = matrixCreationObject.get_train_test_matrix()
            np.save('numpy_data/Jester-Training.npy', trainingMatrix)
            np.save('numpy_data/Jester-Testing.npy', testingMatrix)
        time_start = time.time()
        alsObject = ALS(trainingMatrix, testingMatrix, k, lambdaVal, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("Jester\n")
        StringBuilder.append(alsObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    print("\n\n")
    t4 = time.time()
    total = (t4 - t0) / float(60)
    x = ("\nrun time {0}".format(round(total, 6)))
    print(x)
    StringBuilder.append("\n" + x)
    if data == 5:
        f1 = open('./information/ALS-Results.txt', 'w+')
        f1.write(to_string(StringBuilder))


if __name__ == "__main__":
    main()
