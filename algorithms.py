import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv as inv_
from numpy.linalg import eig, eigvals
from numpy.linalg import det
import math


class RegressionAlgorithms:
    """
    It has the implementation of Vanilla Linear Regression,
    Bayesian Linear Regression and MSE calculate
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.data_len = len(self.data)

    def regularized_LR(self, data_fraction=1.0, lambd=0.0):
        """
        It calculates the weight matrix based on the lambda value
        Data array and label array are returned so as to keep track
        of there shape
        """
        subset_len = int(self.data_len * data_fraction)
        data_list = self.data[:subset_len]
        label_list = self.label[:subset_len]
        data_arr = np.array(data_list)
        label_arr = np.array(label_list)

        identity_mat = np.identity(data_arr.shape[1])
        w = inv_(lambd * identity_mat + data_arr.T @ data_arr) @ data_arr.T @ label_arr
        return w, data_arr, label_arr

    def bayesian_LR(
        self, data_fraction=1.0, alpha=5.0, beta=1.0, tolerance=0.0001
    ):
        """
        It computes the alpha and beta values iteratively based on the tolerance
        Finally it returns the log evidence function and regularization lambda value

        Program flow:
        Calculate M_N and S_N --> Calculate the eigen value of beta*phi.T@phi
        --> Calculate alpha and beta --> Check the convergence condition, if met then exit
        --> else continue updating alpha and beta
        """
        N = int(self.data_len * data_fraction)
        data_list = self.data[:N]
        label_list = self.label[:N]
        data_arr = np.array(data_list)
        M = data_arr.shape[1]
        label_arr = np.array(label_list)
        updated_alpha = 0.0
        updated_beta = 0.0
        iterations = 0
        converge_bool = False

        while not converge_bool:
            S_N_inv = alpha * np.identity(M) + beta * data_arr.T @ data_arr
            S_N = inv_(S_N_inv)
            M_N = (beta * S_N) @ data_arr.T @ label_arr
            lambd_eig = eigvals(beta * data_arr.T @ data_arr)
            gamma = 0
            for i in range(lambd_eig.shape[0]):
                gamma += lambd_eig[i] / (lambd_eig[i] + alpha)
            updated_alpha = gamma / (M_N.T @ M_N)
            updated_beta = (1 / (N - gamma)) * np.sum(
                (label_arr - (data_arr @ M_N)) ** 2
            )
            updated_beta = 1 / updated_beta
            if (
                abs(beta - updated_beta) <= tolerance
                and abs(alpha - updated_alpha) <= tolerance
            ):
                converge_bool = True
            elif iterations == 1000:
                converge_bool = True
            alpha = updated_alpha
            beta = updated_beta
            iterations += 1
        reg_ld = alpha / beta
        log_evidence = (
            M / 2 * (math.log(alpha))
            + N / 2 * (math.log(beta / (2 * math.pi)))
            - (beta / 2) * label_arr.T @ label_arr
            + 0.5 * M_N.T @ inv_(S_N) @ M_N
            + 0.5 * math.log(det(S_N))
        )
        return S_N, M_N, reg_ld, alpha, beta, data_arr, label_arr, log_evidence

    def calculate_MSE(self, test_data, w, test_label):
        test_data_arr = np.array(test_data)
        test_label_arr = np.array(test_label)
        mse = np.sum((test_data_arr @ w - test_label_arr) ** 2)
        return mse / test_data_arr.shape[0]
