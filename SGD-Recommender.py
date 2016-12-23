import MovieLens1MilDataHelper
import MovieLens100kDataHelper
import BookCrossingDataHelper
import JesterDataHelper
import numpy as np
from pathlib import Path
import time


# Stochastic Gradient Descent
class SGD(object):
    def __init__(self, training_matrix, testing_matrix, lambda_val, k, alpha, data_range):
        self.m, self.n = training_matrix.shape
        self.train_matrix, self.test_matrix = training_matrix, testing_matrix
        self.lambda_value, self.k, self.alpha, self.range = lambda_val, k, alpha, data_range
        self.set_non_zero_train(), self.set_non_zero_test()
        self.set_p(), self.set_q()

    @staticmethod
    def prediction(p, q):
        return np.dot(q, p.T)

    def set_non_zero_train(self):
        non_zero_train = self.train_matrix > 0
        non_zero_train[non_zero_train == True] = 1
        non_zero_train[non_zero_train == False] = 0
        self.non_zero_train = non_zero_train

    def set_non_zero_test(self):
        non_zero_test = self.test_matrix > 0
        non_zero_test[non_zero_test == True] = 1
        non_zero_test[non_zero_test == False] = 0
        self.non_zero_test = non_zero_test

    def get_non_zero_train(self):
        return self.non_zero_train

    def get_non_zero_test(self):
        return self.non_zero_test

    def set_p(self):
        self.P = np.random.rand(self.m, self.k) * self.range

    def get_p(self):
        return self.P

    def set_q(self):
        self.Q = np.random.rand(self.n, self.k) * self.range

    def get_q(self):
        return self.Q

    def rmse_train(self, q, p):
        predictionMatrix = self.prediction(p, q).T
        train_matrix, non_zero_train = self.train_matrix, self.non_zero_train
        return np.sqrt(np.sum((non_zero_train * (train_matrix - predictionMatrix)) ** 2) / np.sum(non_zero_train))

    def rmse_test(self, q, p):
        predictionMatrix = self.prediction(p, q).T
        test_matrix, non_zero_test = self.test_matrix, self.non_zero_test
        return np.sqrt(np.sum((non_zero_test * (test_matrix - predictionMatrix)) ** 2) / np.sum(non_zero_test))

    def algorithm(self):
        users, items = self.train_matrix.nonzero()
        prev_train_rmse, prev_testing_rmse = float("Inf"), float("Inf")
        Q, P = self.get_q(), self.get_p()
        learningRate, regularization = self.alpha, self.lambda_value
        epoch = 0
        print("\tSGD")
        while True:
            for u, i in zip(users, items):
                e = self.train_matrix[u, i] - self.prediction(P[u, :], Q[i, :])
                Q[i, :] += learningRate * (e * self.P[u, :] - regularization * self.Q[i, :])
                P[u, :] += learningRate * (e * self.Q[i, :] - regularization * self.P[u, :])
            train_rmse, test_rmse = self.rmse_train(Q, P), self.rmse_test(Q, P)
            if prev_testing_rmse < test_rmse or epoch is 10:
                break
            print("\t\tepoch {0} \tTraining-RMSE {1} \tTesting-RMSE {2}".format(round(epoch, 2),
                                                                                round(train_rmse, 5),
                                                                                round(test_rmse, 5)))
            prev_train_rmse = train_rmse
            prev_testing_rmse = test_rmse
            self.P = P
            self.Q = Q
            epoch += 1
        x = ('\tFinal Training-RMSE {0}\n\tFinal Testing-RMSE {1}\n'.format(round(prev_train_rmse, 5),
                                                                            round(prev_testing_rmse, 5)))
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
    print("Stochastic Gradient Descent")
    data = int(input("Enter file\n1 for MovieLens\n2 for Book-Crossing\n3 for Jester\n4 for all\n"))
    if data == 1 or data == 4 or data == 5:
        print("MovieLens")
        lambdaValue, k, alpha, data_range = 0, 0, 0, 0
        movieLensIndicator = int(input("Enter\n1 for MovieLens100k\n2 for MovieLens1Millon\n"))
        if data != 5:
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaValue = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            alpha_input = float(input("\tEnter the learning rate value (0) for default value\n\t"))
            alpha = .001 if alpha_input == 0 else alpha_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 2 if data_range_input == 0 else data_range_input
        if data == 5:
            lambdaValue = .1
            k = 3
            alpha = .001
            data_range = 2
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
        sgdObject = SGD(trainingMatrix, testingMatrix, lambdaValue, k, alpha, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("MovieLens\n")
        StringBuilder.append(sgdObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))

    if data == 2 or data == 4 or data == 5:
        print("Book-Crossing")
        lambdaValue, k, alpha, data_range = 0, 0, 0, 0
        if data != 5:
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaValue = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            alpha_input = float(input("\tEnter the learning rate value (0) for default value\n\t"))
            alpha = .01 if alpha_input == 0 else alpha_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 3 if data_range_input == 0 else data_range_input
        if data == 5:
            lambdaValue = .1
            k = 3
            alpha = .01
            data_range = 3
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
        sgdObject = SGD(trainingMatrix, testingMatrix, lambdaValue, k, alpha, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("Book-Crossing\n")
        StringBuilder.append(sgdObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))

    if data == 3 or data == 4 or data == 5:
        print("Jester")
        lambdaValue, k, alpha, data_range = 0, 0, 0, 0
        if data != 5:
            lambda_input = float(input("\tEnter the regularization weight (0) for default value\n\t"))
            lambdaValue = .1 if lambda_input == 0 else lambda_input
            k_input = int(input("\tEnter the low rank matrix approximation value (0) for default value\n\t"))
            k = 3 if k_input == 0 else k_input
            alpha_input = float(input("\tEnter the learning rate value (0) for default value\n\t"))
            alpha = .0001 if alpha_input == 0 else alpha_input
            data_range_input = int(input("\tEnter the data_range initialization value (0) for default value\n\t"))
            data_range = 3 if data_range_input == 0 else data_range_input
        if data == 5:
            lambdaValue = .1
            k = 3
            alpha = .0001
            data_range = 3
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
        sgdObject = SGD(trainingMatrix, testingMatrix, lambdaValue, k, alpha, data_range)
        print("\n\t---------------------------------------------------------------")
        StringBuilder.append("Jester\n")
        StringBuilder.append(sgdObject.algorithm())
        time_end = time.time()
        total = (time_end - time_start) / float(60)
        print("\nrun time {0}".format(round(total, 6)))
    print("\n\n")
    t4 = time.time()
    total = (t4 - t0) / float(60)
    x = ("\nrun-time {0}".format(round(total, 6)))
    print(x)
    StringBuilder.append("\n" + x)

    if data == 5:
        f1 = open('./information/SGD-Results.txt', 'w+')
        f1.write(to_string(StringBuilder))


if __name__ == "__main__":
    main()
