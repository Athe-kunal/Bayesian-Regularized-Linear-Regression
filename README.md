# COMPARISON BETWEEN VANILLA AND BAYESIAN LINEAR REGRESSION

## This repository covers three tasks, (i) compares between bayesian and vanilla linear regression, (ii)compares the regularization parameter,(iii) applies linear regression of a multinomial polynomial.

### There are three important files here `algorithms.py`, which has the bayesian and vanilla linear regression algorithm. `task_1.py` and `task_2.py` are the plotting and implementation files for Task 1 and Task 2 respectively.

Run this for Task 1 
```
python task_1.py
```
### Outputs:
#### It will output 6 plots: `CRIME_MODEL_SELECTION.png` and `HOUSING_MODEL_SELECTION.png` which has the plots for trend of alphas, betas and test set error. Then `HOUSING ERRORS.jpg` and `CRIME ERRORS.jpg` compares between vanilla and bayesian linear regression test set errors. Then it produces two plots: `CRIME_MLE_LAMBDA.jpg` and `HOUSING_MLE_LAMBDA.jpg` which has the plots for test set errors for different values of lambdas.


#### Run this for Task 2 
```
python task_2.py
```
### Outputs:
#### It outputs 2 plots: `f3.jpg` and `f5.jpg`, which has the log evidence and test set error for different degrees of polynomial.

