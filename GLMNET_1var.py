import time
import torch
import numpy as np
torch.set_printoptions(sci_mode=False)
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def p_norm(v, p):
    # Compute the p-norm of a vector v
    v = torch.abs(v)
    return v.pow(p).sum()

def predict(x, w, b):
    # Make predictions using a linear model
    return torch.matmul(x, w) + b


def MSE(y_pred, y):
    # Compute the mean squared error between predicted and actual values
    return (y_pred - y).pow(2).sum()


def error_PQ_GLMNET(x, w, b, y, alpha, lamda, p, q):
    # Compute the total error of the model, including mean squared error and regularization penalty
    y_pred = predict(x, w, b)
    x_t = torch.transpose(x, 0, 1)
    error = (MSE(y_pred, y)/(2*len(x_t))) + lamda * (alpha*p_norm(w,p)/p + (1-alpha)*p_norm(w,q)/q)
    return error


def classify(x):
    # Classify the elements of x as 0 or 1 based on a threshold of 0.5
    for i in range(len(x)):
        if x[i] < 0.5:
            x[i] = 0
        else:
            x[i] = 1
    return x


def logLoss(y, y_pred):
    # Compute the logarithmic loss between predicted and actual values
    eps = 1e-40
    return -(1/len(y_pred)) * (y * torch.log(y_pred + eps) + (1 - y) * torch.log(1 - y_pred + eps)).sum()


def error_PQ_GLMNET_Binary(x, w, b, y, alpha, lamda, p, q):
    # Compute the total error of the binary classification model, including log loss and regularization penalty
    y_pred = torch.sigmoid(predict(x, w, b))
    error = logLoss(y_pred, y) + lamda * (alpha*p_norm(w,p)/p + (1-alpha)*p_norm(w,q)/q)
    return error


def ind_OneVar_glmnet(x, y, alpha, lamda, w=None, b=None, max_iterations=50,
                      learning_rate=0.01, verbose=True, threshold_error=1e-6,
                      model_type='Cont', p=1, q=1):
    # Perform individual variable (univariate) glmnet optimization
    torch.manual_seed(42)

    if w is None:
        w = torch.zeros(x.shape[1], requires_grad=True)
    if b is None:
        b = torch.zeros(1, requires_grad=True)

    epoch = 0

    w_prev = torch.zeros(len(w))

    while True:
        epoch += 1

        if model_type == 'Cont':
            loss = error_PQ_GLMNET(x, w, b, y, alpha=alpha, lamda=lamda, p=p, q=q)
            loss.backward()
        elif model_type == 'Bin':
            loss = error_PQ_GLMNET_Binary(x, w, b, y, alpha=alpha, lamda=lamda, p=p, q=q)
            loss.backward()

        with torch.no_grad():
            w_update = learning_rate*w.grad
            b_update = learning_rate*b.grad

            w -= w_update
            b -= b_update

            w.grad = None
            b.grad = None

            learning_rate = learning_rate * 0.999

            if verbose:
                if model_type == 'Cont':
                    print("Epoch = ", epoch, "/", max_iterations, " Error = ",
                          torch.sqrt(MSE(y, torch.matmul(x, w) + b)).item(),
                          " Max w dif = ", torch.max(torch.abs(w - w_prev)).item())
                elif model_type == 'Bin':
                    print("Epoch = ", epoch, "/", max_iterations, " Error = ",
                          logLoss(y, torch.sigmoid(b + torch.matmul(x, w))).item())

        if epoch >= max_iterations or torch.max(torch.abs(w - w_prev)) < threshold_error:
            break
        w_prev = w.clone()

    def predict(x):
        if model_type == 'Cont':
            return b + torch.matmul(x, w)
        elif model_type == 'Bin':
            return classify(torch.sigmoid(b + torch.matmul(x, w)))

    class Result:
        def __init__(self, w, b, predict):
            self.w = w
            self.b = b
            self.predict = predict

    return Result(w, b, predict)


def std_OneVar_glmnet(x_train, y_train, x_validation, y_validation, alpha, lamda_path,
                      threshold_error=1e-6, max_iterations=50, learning_rate=0.01, verbose=True,
                      model_type='Cont', p=1, q=1):
    # Perform standard glmnet optimization with univariate feature selection
    torch.manual_seed(42)
    dtype = torch.float
    device = torch.device("cpu")

    x_t = torch.transpose(x_train, 0, 1)
    n = len(x_t)

    w = torch.zeros(n, dtype=dtype, device=device, requires_grad=True)
    b = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)

    lamda_max = max(lamda_path)
    lamda_opt = lamda_max

    w_dict = {}
    b_dict = {}
    error_dict = {}

    for lamda in lamda_path:
        if verbose:
            print("lamda = ", lamda)
            print("----------------------")

        result = ind_OneVar_glmnet(x_train, y_train, alpha, lamda,
                                   w.clone().detach().requires_grad_(True),
                                   b.clone().detach().requires_grad_(True),
                                   max_iterations, learning_rate, verbose,
                                   threshold_error, model_type, p, q)

        w = result.w.clone().detach()
        b = result.b.clone().detach()

        if model_type == 'Cont':
            loss = torch.sqrt(MSE(y_validation, torch.matmul(x_validation, w) + b)).item()
        elif model_type == 'Bin':
            loss = logLoss(y_validation, torch.sigmoid(b + torch.matmul(x_validation, w))).item()

        w_dict[lamda] = w
        b_dict[lamda] = b
        error_dict[lamda] = loss

        if loss < error_dict[lamda_opt]:
            lamda_opt = lamda

    class Result:
        def __init__(self, w_dict, b_dict, error_dict, lamda_opt, lamda_max):
            self.w_dict = w_dict
            self.b_dict = b_dict
            self.error_dict = error_dict
            self.lamda_opt = lamda_opt
            self.w_opt = w_dict[lamda_opt]
            self.b_opt = b_dict[lamda_opt]

    return Result(w_dict, b_dict, error_dict, lamda_opt, lamda_max)


def cv_OneVar_glmnet(x, y, alpha, threshold_error=1e-5, max_iterations=25,
                     learning_rate=0.01, lambdas=100, k=5, verbose=True,
                     model_type='Cont', p=1, q=1):
    # Perform cross-validation for glmnet with univariate feature selection
    start_time = time.time()
    validation_errors = {}

    kf = KFold(n_splits=k, shuffle=True)
    w_folds = {}
    b_folds = {}

    i = 0

    x_t = torch.transpose(x, 0, 1)

    lamda_max = max([torch.matmul(x_t[i], y - y.mean()).abs() for i in range(len(x_t))]) / (len(x_t)*alpha)

    lamda_min = 0.001 * lamda_max
    path = (np.power(10, np.linspace(np.log10(lamda_min), np.log10(lamda_max), lambdas)))[::-1]

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]

        ret = std_OneVar_glmnet(x_train, y_train, x_val, y_val,
                                alpha=alpha, lamda_path=path,
                                threshold_error=threshold_error,
                                max_iterations=max_iterations,
                                learning_rate=learning_rate,
                                verbose=verbose,
                                model_type=model_type, p=p, q=q)

        w_folds[i] = ret.w_dict
        b_folds[i] = ret.b_dict
        validation_errors[i] = ret.error_dict

        i += 1

    class Result:
        def __init__(self, validation_errors, path, w_folds, b_folds, lamda_max, start_time):
            self.validation_errors = validation_errors
            self.path = path
            self.w_folds = w_folds
            self.b_folds = b_folds
            self.lamda_max = lamda_max
            self.start_time = start_time

            self.w = {}
            self.b = {}
            self.validation_errors_means = {}
            self.validation_errors_sd = {}

            for key in w_folds[0].keys():
                self.w[key] = torch.zeros(len(x_t), dtype=torch.float, device=torch.device("cpu"))
                self.b[key] = torch.zeros(1, dtype=torch.float, device=torch.device("cpu"))
                for i in range(k):
                    self.w[key] += w_folds[i][key]
                    self.b[key] += b_folds[i][key]
                self.w[key] = self.w[key] / k
                self.b[key] = self.b[key] / k

            for key in validation_errors[0].keys():
                self.validation_errors_means[key] = 0
                self.validation_errors_sd[key] = 0
                error_list = []
                for i in range(k):
                    self.validation_errors_means[key] += validation_errors[i][key]
                    error_list.append(validation_errors[i][key])
                self.validation_errors_means[key] = self.validation_errors_means[key] / k
                self.validation_errors_sd[key] = np.std(error_list) / np.sqrt(len(path))

            self.lambda_opt = min(self.validation_errors_means, key=self.validation_errors_means.get)

            self.w_opt = self.w[self.lambda_opt]
            self.b_opt = self.b[self.lambda_opt]

            intervalo = (self.validation_errors_means[self.lambda_opt] - self.validation_errors_sd[self.lambda_opt],
                         self.validation_errors_means[self.lambda_opt] + self.validation_errors_sd[self.lambda_opt])

            self.lambda_1se = max([k for k in self.validation_errors_means.keys()
                                   if intervalo[0] <= self.validation_errors_means[k] <= intervalo[1]],
                                  key=self.validation_errors_means.get)
            self.w_1se = self.w[self.lambda_1se]
            self.b_1se = self.b[self.lambda_1se]

            def predict_opt(x):
                if model_type == 'Cont':
                    return torch.matmul(x, self.w_opt) + self.b_opt
                elif model_type == 'Bin':
                    return classify(torch.sigmoid(self.b_opt + torch.matmul(x, self.w_opt)))

            def predict_1se(x):
                if model_type == 'Cont':
                    return torch.matmul(x, self.w_1se) + self.b_1se
                elif model_type == 'Bin':
                    return classify(torch.sigmoid(self.b_1se + torch.matmul(x, self.w_1se)))

            self.predict_opt = predict_opt
            self.predict_1se = predict_1se

            self.validation_errors_means_low = [self.validation_errors_means[key] - 1.96*self.validation_errors_sd[key]
                                                for key in self.validation_errors_means.keys()]
            self.validation_errors_means_high = [self.validation_errors_means[key] + 1.96*self.validation_errors_sd[key]
                                                 for key in self.validation_errors_means.keys()]

            def error_evolution():
                plt.figure()
                plt.title('Validation error evolution')
                plt.fill_between(path, self.validation_errors_means_low, self.validation_errors_means_high,
                                 color='blue', alpha=0.2)
                plt.semilogx(path, self.validation_errors_means.values(), lw=1, color='blue', label='Error')
                plt.axvline(x=self.lambda_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambda_1se, color='green', label='lambda_1se')

                plt.xticks([min((path)), (self.lambda_opt), (self.lambda_1se), max((self.path))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend(prop={'size': 8})

            def coefficients_evolution():
                plt.figure()
                plt.title('Coefficients evolution')

                transposed_dict = {}

                for key, value in self.w.items():
                    for i, v in enumerate(value):
                        transposed_dict[i] = transposed_dict.get(i, []) + [v.item()]

                for key in transposed_dict.keys():
                    plt.semilogx(path, transposed_dict[key], lw=1, label='w_' + str(key))

                plt.axvline(x=self.lambda_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambda_1se, color='green', label='lambda_1se')

                plt.xticks([min((path)), (self.lambda_opt), (self.lambda_1se), max((path))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend(prop={'size': 8}, bbox_to_anchor=(1.05, 1), loc='upper left')

            self.error_evolution = error_evolution
            self.coefficients_evolution = coefficients_evolution
            self.time = time.time() - start_time

    return Result(validation_errors, path, w_folds, b_folds, lamda_max, start_time)





