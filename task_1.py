from algorithms import RegressionAlgorithms
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 20)
plt.style.use("ggplot")

# Reading data
def read_data(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()
    all_data = []
    for line in lines:
        line = line[:-1]
        line = [float(k) for k in line.split(",")]
        all_data.append(line)
    return all_data


train_crime = read_data("pp2data\\train-crime.csv")
train_housing = read_data("pp2data\\train-housing.csv")

test_crime = read_data("pp2data\\test-crime.csv")
test_housing = read_data("pp2data\\test-housing.csv")

# Read label
def read_label(file_name: str):
    with open(file_name, "r") as f:
        lines = f.readlines()
    all_labels = []
    for line in lines:
        line = line[:-1]
        line = [float(k) for k in line.split(",")]
        all_labels.append(line[0])
    return all_labels


train_crime_label = read_label("pp2data\\trainR-crime.csv")
train_housing_label = read_label("pp2data\\trainR-housing.csv")

test_crime_label = read_label("pp2data\\testR-crime.csv")
test_housing_label = read_label("pp2data\\testR-housing.csv")

# Run the model selection algorithm
def model_selection(train_data_fractions: list):
    # Crime data
    crimeRegression = RegressionAlgorithms(train_crime, train_crime_label)
    crime_alphas = []
    crime_betas = []
    crime_lambdas = []
    crime_test_set_errors = []

    print("For CRIME DATA")
    for data_fraction in train_data_fractions:
        (
            S_N,
            M_N,
            regularization_lambda,
            alpha,
            beta,
            data_arr,
            _,
            _,
        ) = crimeRegression.bayesian_LR(data_fraction)
        w_bayes, data_arr, _ = crimeRegression.regularized_LR(
            data_fraction, regularization_lambda
        )
        test_mse = crimeRegression.calculate_MSE(test_crime, w_bayes, test_crime_label)
        crime_alphas.append(alpha)
        crime_betas.append(beta)
        crime_lambdas.append(regularization_lambda)
        crime_test_set_errors.append(test_mse)
        print(f"For train set size of {data_arr.shape[0]} the alpha is {alpha}")
        print(f"For train set size of {data_arr.shape[0]} the beta is {beta}")
        print(
            f"For train set size of {data_arr.shape[0]} the lambda is {regularization_lambda}"
        )
        print("-" * 100)

    # Housing data
    housingRegression = RegressionAlgorithms(train_housing, train_housing_label)
    housing_alphas = []
    housing_betas = []
    housing_lambdas = []
    housing_test_set_errors = []
    print("For HOUSING DATA")
    for data_fraction in train_data_fractions:
        (
            _,
            M_N,
            regularization_lambda,
            alpha,
            beta,
            data_arr,
            _,
            _,
        ) = housingRegression.bayesian_LR(data_fraction, tolerance=0.00001)
        w_bayes, _, _ = housingRegression.regularized_LR(
            data_fraction, regularization_lambda
        )
        test_mse = housingRegression.calculate_MSE(
            test_housing, w_bayes, test_housing_label
        )
        housing_alphas.append(alpha)
        housing_betas.append(beta)
        housing_lambdas.append(regularization_lambda)
        housing_test_set_errors.append(test_mse)
        print(f"For train set size of {data_arr.shape[0]} the alpha is {alpha}")
        print(f"For train set size of {data_arr.shape[0]} the beta is {beta}")
        print(
            f"For train set size of {data_arr.shape[0]} the lambda is {regularization_lambda}"
        )
        print("-" * 100)

    return (
        crime_alphas,
        crime_betas,
        crime_lambdas,
        crime_test_set_errors,
        housing_alphas,
        housing_betas,
        housing_lambdas,
        housing_test_set_errors,
    )


def MLE(train_data_fractions: list, lambd=0.0):
    # Maximum Log likelihood function
    crimeRegression = RegressionAlgorithms(train_crime, train_crime_label)
    houseRegression = RegressionAlgorithms(train_housing, train_housing_label)
    crime_test_errors = []
    house_test_errors = []
    for data_fraction in train_data_fractions:
        w_crime, _, _ = crimeRegression.regularized_LR(data_fraction, lambd)
        crime_test_mse = crimeRegression.calculate_MSE(
            test_crime, w_crime, test_crime_label
        )
        w_house, _, _ = houseRegression.regularized_LR(data_fraction, lambd)
        house_test_mse = houseRegression.calculate_MSE(
            test_housing, w_house, test_housing_label
        )
        crime_test_errors.append(crime_test_mse)
        house_test_errors.append(house_test_mse)
    return crime_test_errors, house_test_errors


def plot_model_selection(
    alphas: list,
    betas: list,
    lambdas: list,
    test_set_errors: list,
    dataset_sizes: list,
    title: str,
):

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    fig.suptitle(title)
    ax1.plot(dataset_sizes, alphas, marker="o", label="Alphas", linestyle="dotted")
    ax1.set_title("Alphas")
    ax2.plot(dataset_sizes, betas, marker="s", label="Betas", linestyle="dotted")
    ax2.set_title("Betas")
    ax3.plot(dataset_sizes, lambdas, marker="o", label="Lambdas", linestyle="dotted")
    ax3.set_title("Lambdas")
    ax4.plot(
        dataset_sizes,
        test_set_errors,
        marker="s",
        label="Test Set Error",
        linestyle="dotted",
    )
    ax4.set_title("Test Set Error")
    ax4.set_xlabel("Training Set Size")
    ax4.set_ylim(0, 1)
    plt.savefig(f"{title}.jpg", bbox_inches="tight")
    plt.show()


def plot_errors(
    test_errors: list, bayes_test_errors: list, dataset_sizes: list, title: str
):
    fig, (ax1) = plt.subplots(1)
    ax1.set_title(title)
    ax1.plot(dataset_sizes, test_errors, marker="o", linestyle="dotted", label="MLE")
    ax1.plot(
        dataset_sizes,
        bayes_test_errors,
        marker="o",
        linestyle="dotted",
        label="Bayesian LR",
    )
    ax1.legend(loc="best")
    ax1.set_ylim(0, 1)
    fig.savefig(f"{title}.jpg", bbox_inches="tight")
    plt.show()


def plot_lambdas(lambdas_dict: dict, dataset_sizes: list, bayesian_error:list, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    set_title = title.split("_")[0]
    print(f"For {set_title}")
    print(lambdas_dict)

    for lambd in lambdas_dict:
        ax.plot(
            dataset_sizes,
            lambdas_dict[lambd],
            label=f"{lambd}",
            linestyle="dotted",
            marker="o",
        )
    ax.plot(dataset_sizes,bayesian_error,label='Bayesian',linestyle='dotted',marker='s')
    ax.set_title(f"{set_title} DATA WITH VARYING LAMBDA")
    ax.set_xlabel("Dataset Sizes")
    ax.set_ylabel("Test Set Error")
    ax.legend(loc="best")
    fig.savefig(f"{title}.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Question 1
    train_data_fractions = np.linspace(0.1, 1.0, 10)
    crime_train_sizes = [
        len(train_crime) * data_fractions for data_fractions in train_data_fractions
    ]
    house_train_sizes = [
        len(train_housing) * data_fractions for data_fractions in train_data_fractions
    ]

    (
        crime_alphas,
        crime_betas,
        crime_lambdas,
        crime_test_set_errors,
        housing_alphas,
        housing_betas,
        housing_lambdas,
        housing_test_set_errors,
    ) = model_selection(train_data_fractions)
    plot_model_selection(
        crime_alphas,
        crime_betas,
        crime_lambdas,
        crime_test_set_errors,
        crime_train_sizes,
        "CRIME_MODEL_SELECTION",
    )
    plot_model_selection(
        housing_alphas,
        housing_betas,
        housing_lambdas,
        housing_test_set_errors,
        house_train_sizes,
        "HOUSING_MODEL_SELECTION",
    )

    # ---------------------------------------------------------------
    # Question 2
    crime_test_errors, house_test_errors = MLE(train_data_fractions)
    plot_errors(
        crime_test_errors,
        crime_test_set_errors,
        crime_train_sizes,
        "CRIME ERRORS REGULARIZED VS NON REGULARIZED",
    )
    plot_errors(
        house_test_errors, housing_test_set_errors, house_train_sizes, "HOUSING ERRORS"
    )

    # ------------------------------------------------------------
    # Question 3

    lambdas = [1.0, 33.0, 100.0, 1000.0]
    crime_lambdas_dict = dict.fromkeys(lambdas, 0)
    housing_lambdas_dict = dict.fromkeys(lambdas, 0)
    for lambd in lambdas:
        crime_test_errors, house_test_errors = MLE(train_data_fractions, lambd=lambd)
        crime_lambdas_dict[lambd] = crime_test_errors
        housing_lambdas_dict[lambd] = house_test_errors
    # plotting for CRIME dataset
    plot_lambdas(crime_lambdas_dict, crime_train_sizes,crime_test_set_errors, "CRIME_MLE_LAMBDA")
    # plotting for HOUSING dataset
    plot_lambdas(housing_lambdas_dict, house_train_sizes,housing_test_set_errors, "HOUSING_MLE_LAMBDA")
