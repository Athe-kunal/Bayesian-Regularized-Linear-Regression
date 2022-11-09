from algorithms import RegressionAlgorithms
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
plt.style.use("ggplot")


def read_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    lines = [float(data[:-1]) for data in lines]
    return lines


f3_train = read_data("pp2data\\train-f3.csv")
f5_train = read_data("pp2data\\train-f5.csv")
f3_test = read_data("pp2data\\test-f3.csv")
f5_test = read_data("pp2data\\test-f5.csv")

f3_train_label = read_data("pp2data\\trainR-f3.csv")
f3_test_label = read_data("pp2data\\testR-f3.csv")
f5_train_label = read_data("pp2data\\trainR-f5.csv")
f5_test_label = read_data("pp2data\\testR-f5.csv")

f3_train_label = np.array(f3_train_label, dtype=np.float64)
f3_test_label = np.array(f3_test_label, dtype=np.float64)
f5_train_label = np.array(f5_train_label, dtype=np.float64)
f5_test_label = np.array(f5_test_label, dtype=np.float64)

# It converts the dataset into a 10 degree polynomial dataset
# Then as per the condition, we select the required number of polynomials
def convert_to_all_data(dataset: list, degree):
    converted_dataset = []
    for data in dataset:
        single_dataset = []
        for d in range(degree + 1):
            single_dataset.append(data**d)
        converted_dataset.append(single_dataset)
    converted_dataset = np.array(converted_dataset, dtype=np.float64)
    return converted_dataset


def get_req_data(dataset, d):
    return np.array([data[: d + 1] for data in dataset])


def plot_evidences_errors(
    evidence_list: list,
    mle_error_list: list,
    map_error_list: list,
    degrees: list,
    title: str,
):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f"For dataset {title}")
    ax1.plot(degrees, evidence_list, marker="o", label="Evidence", linestyle="dotted")
    ax1.set_ylabel("Log Evidence")
    ax2.plot(degrees, map_error_list, marker="s", label="Bayesian", linestyle="dotted")
    ax2.plot(degrees, mle_error_list, marker="o", label="MLE", linestyle="dotted")
    ax2.set_ylabel("Test set Error")
    ax2.set_xlabel("Degree")
    ax2.legend(loc="best")
    fig.savefig(f"{title}.jpg", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    degrees = [i for i in range(1, 11)]
    f3_train = convert_to_all_data(f3_train, degrees[-1])
    f5_train = convert_to_all_data(f5_train, degrees[-1])
    f3_test = convert_to_all_data(f3_test, degrees[-1])
    f5_test = convert_to_all_data(f5_test, degrees[-1])

    f3_log_evidences = []
    f3_mle_test_mse = []
    f3_map_test_mse = []

    f5_log_evidences = []
    f5_mle_test_mse = []
    f5_map_test_mse = []

    for d in degrees:
        f3_train_d = get_req_data(f3_train, d)
        f5_train_d = get_req_data(f5_train, d)
        f3_test_d = get_req_data(f3_test, d)
        f5_test_d = get_req_data(f5_test, d)
        # print(f'Computing for {f3_train_d.shape} and {f5_train_d.shape}')
        f3 = RegressionAlgorithms(f3_train_d, f3_train_label)
        w_f3, _, _ = f3.regularized_LR()
        MLE_mse = f3.calculate_MSE(f3_test_d, w_f3, f3_test_label)
        f3_mle_test_mse.append(MLE_mse)
        (
            S_N,
            M_N_f3,
            regularization_lambda,
            alpha,
            beta,
            _,
            _,
            log_evidence,
        ) = f3.bayesian_LR()
        M_N_f3 = np.real(M_N_f3)
        # Calculating the MSE values using M_N (MAP estimate)
        MAP_mse = f3.calculate_MSE(f3_test_d, M_N_f3, f3_test_label)
        f3_map_test_mse.append(MAP_mse)
        f3_log_evidences.append(log_evidence)

        f5 = RegressionAlgorithms(f5_train_d, f5_train_label)
        w_f5, _, _ = f5.regularized_LR()
        MLE_mse = f5.calculate_MSE(f5_test_d, w_f5, f5_test_label)
        f5_mle_test_mse.append(MLE_mse)
        (
            S_N,
            M_N_f5,
            regularization_lambda,
            alpha,
            beta,
            data_arr,
            label_arr,
            log_evidence,
        ) = f5.bayesian_LR()
        M_N_15 = np.real(M_N_f5)
        MAP_mse = f5.calculate_MSE(f5_test_d, M_N_f5, f5_test_label)
        f5_map_test_mse.append(MAP_mse)
        f5_log_evidences.append(log_evidence)
    # print(f3_log_evidences,f3_mle_test_mse,f3_map_test_mse)
    # print(f5_log_evidences,f5_mle_test_mse,f5_map_test_mse)
    print('F3 MAP Test Errors')
    print(f3_map_test_mse)
    print('F3 MLE Test Errors')
    print(f3_mle_test_mse)

    print('F5 MAP Test Errors')
    print(f5_map_test_mse)
    print('F5 MLE Test Errors')
    print(f5_mle_test_mse)

    plot_evidences_errors(
        f3_log_evidences, f3_mle_test_mse, f3_map_test_mse, degrees, "f3"
    )
    plot_evidences_errors(
        f5_log_evidences, f5_mle_test_mse, f5_map_test_mse, degrees, "f5"
    )
