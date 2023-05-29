# AutomaticDifferentiation-ElasticNet

This project aims to be a reimplementation of the Elastic Net algorithm using the automatic differentiation engine provided by the PyTorch library. The code implements linear regression, binary regression, and multivariable regression with one binary output and the other continuous.


# Prerequisites
The following libraries are required to run the code:

+ torch: PyTorch library for tensor computations
+ numpy: Numerical computing library
+ scikit-learn: Machine learning library for model selection
+ matplotlib: Plotting library

# Usage
The code provides several functions for training and evaluating GLMNET models. Here's an overview of the available functions:

# cv_GLMNET

This function performs cross-validation using GLMNET algorithm. It takes the following parameters:

+ x: Input features (torch tensor)
+ y: Target values (torch tensor)
+ alpha: Mixing parameter between Lp and Lq regularization (float)
+ umbral_error: Threshold for convergence criterion (float, default: 1e-100)
+ max_iteraciones: Maximum number of iterations (int, default: 25)
+ learning_rate: Learning rate for gradient descent (float, default: 0.01)
+ lambdas: Number of lambda values to try (int, default: 100)
+ k: Number of cross-validation folds (int, default: 5)
+ verbose: Whether to print progress information (bool, default: True)
+ tipo: Type of problem, either 'Continuo' or 'Binario' (str, default: 'Continuo')
+ p: Lp norm to use for regularization (float, default: 1)
+ q: Lq norm to use for regularization (float, default: 1)


The function returns an object with the following attributes:

+ errores: List of validation errors for each lambda value (list)
+ lambdas: List of lambda values (list)
+ w: Averaged weights across cross-validation folds (list)
+ b: Averaged biases across cross-validation folds (list)
+ lamda_max: Maximum lambda value (float)
+ nulos: Number of zero weights in the model (list)
+ errores_medios: Mean validation error for each lambda value (list)
+ desviaciones: Standard deviation of validation error for each lambda value (list)
+ lambda_opt: Optimal lambda value (float)
+ error_minimo: Minimum validation error (float)
+ w_opt: Weights corresponding to the optimal lambda value (tensor)
+ w_1se: Weights corresponding to the 1SE rule (tensor)
+ b_opt: Bias corresponding to the optimal lambda value (tensor)
+ b_1se: Bias corresponding to the 1SE rule (tensor)
+ f_opt: Function to predict output using optimal lambda (function)
+ f_1se: Function to predict output using 1SE rule (function)
+ errores_medios_bajos: Lower bound of mean validation error (list)
+ errores_medios_altos: Upper bound of mean validation error (list)


You can use the f_opt or f_1se functions to make predictions on new data using the trained model.
For using classic Lasso regression you should use alpha = 1, p = 1
For using classic Lasso regression you should use alpha = 0, q = 2
For using classic ElasticNet regression you should use p = 1, q = 2 (alpha is recommended to be between 0.8 and 0.999)

# cv_GLMNET_Multiple






