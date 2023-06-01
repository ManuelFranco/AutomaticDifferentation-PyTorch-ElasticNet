import time
import torch
import numpy as np
from matplotlib import cm

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



def isActive(w, eps):
    return torch.abs(w) > eps

def error_PQ_GLMNET_Dis(x, w, mask, b, y, alpha, lamda, p, q):
    # Compute the total error of the model, including mean squared error and regularization penalty
    y_pred = predict(x, w, b)
    x_t = torch.transpose(x, 0, 1)
    mask = w*mask
    error = MSE(y_pred, y)/(2*len(x_t)) + lamda * (alpha*(p_norm(w,p)/p + p_norm(mask, p)/p ) + (1-alpha)*(p_norm(w,q)/q  + p_norm(mask, q)/q ))
    return error

def error_PQ_GLMNET_Binary_Dis(x, w, mask, b, y, alpha, lamda, p, q):
    # Compute the total error of the binary classification model, including log loss and regularization penalty
    y_pred = torch.sigmoid(predict(x, w, b))
    mask = w*mask
    error = logLoss(y_pred, y) + lamda * (alpha*(p_norm(w,p)/p + p_norm(mask, p)/p ) + (1-alpha)*(p_norm(w,q)/q  + p_norm(mask, q)/q ))
    return error


def ind_TwoVar_glmnet(x, yCont, yBin,
                      alpha, lamdaCont, lamdaBin,
                      wCont=None, bCont=None, wBin = None, bBin = None,
                      max_iterations=50,
                      learning_rate=0.01, verbose=True, threshold_error=1e-6,
                      p=1, q=1, areIndependent=False):
    torch.manual_seed(42)

    if wCont is None:
        wCont = torch.zeros(x.shape[1], requires_grad=True)
    if bCont is None:
        bCont = torch.zeros(1, requires_grad=True)

    if wBin is None:
        wBin = torch.zeros(x.shape[1], requires_grad=True)
    if bBin is None:
        bBin = torch.zeros(1, requires_grad=True)

    epoch = 0

    wCont_prev = torch.zeros(len(wCont))
    wBin_prev = torch.zeros(len(wBin))

    while True:
        epoch += 1

        if areIndependent:
            maskBoth = isActive(wCont, 0.1) & isActive(wBin, 0.1)
            lossCont = error_PQ_GLMNET_Dis(x, wCont, maskBoth, bCont, yCont,
                                           alpha=alpha, lamda=lamdaCont, p=p, q=q)
            lossBin = error_PQ_GLMNET_Binary_Dis(x, wBin, maskBoth, bBin, yBin,
                                                 alpha=alpha, lamda=lamdaBin, p=p, q=q)
        else:
            lossCont = error_PQ_GLMNET(x, wCont, bCont, yCont,
                                       alpha=alpha, lamda=lamdaCont, p=p, q=q)
            lossBin = error_PQ_GLMNET_Binary(x, wBin, bBin, yBin,
                                             alpha=alpha, lamda=lamdaBin, p=p, q=q)

        lossCont.backward()
        lossBin.backward()

        with torch.no_grad():
            wCont_update = learning_rate*wCont.grad
            bCont_update = learning_rate*bCont.grad

            wBin_update = learning_rate*wBin.grad
            bBin_update = learning_rate*bBin.grad

            wCont -= wCont_update
            bCont -= bCont_update
            wBin -= wBin_update
            bBin -= bBin_update

            wCont.grad = None
            bCont.grad = None
            wBin.grad = None
            bBin.grad = None

            learning_rate = learning_rate * 0.999

            if verbose:
                print("Epoch = ", epoch, "/", max_iterations, " ECont = ",
                      torch.sqrt(MSE(yCont, torch.matmul(x, wCont) + bCont)).item(),
                      " wContDif = ", torch.max(torch.abs(wCont - wCont_prev)).item()
                      , " EBin = ",
                      logLoss(yBin, torch.sigmoid(bBin + torch.matmul(x, wBin))).item(),
                      "wBinDif = ", torch.max(torch.abs(wBin - wBin_prev)).item())

        if ((epoch >= max_iterations) or
                ((torch.max(torch.abs(wCont - wCont_prev)) < threshold_error) and
                 (torch.max(torch.abs(wBin - wBin_prev)) < threshold_error))):
            print( torch.max((wCont - wCont_prev)), torch.max((wBin - wBin_prev)))
            break

        wCont_prev = wCont.clone()
        wBin_prev = wBin.clone()

    def predict(x):
        return (torch.matmul(x, wCont) + bCont, classify(torch.sigmoid(bBin + torch.matmul(x, wBin))))

    class Result:
        def __init__(self, wCont, bCont, wBin, bBin, predict):
            self.wCont = wCont
            self.bCont = bCont
            self.wBin = wBin
            self.bBin = bBin
            self.predict = predict

    return Result(wCont, bCont, wBin, bBin, predict)



def std_TwoVar_glmnet(x_train, x_validation,
                      yCont_train, yCont_validation,
                      yBin_train, yBin_validation,
                      alpha, lamdaCont_path, lamdaBin_path,
                      threshold_error=1e-6, max_iterations=50, learning_rate=0.01, verbose=True,
                      p=1, q=1, areIndependent=False):
    # Perform standard glmnet optimization with univariate feature selection
    torch.manual_seed(42)
    dtype = torch.float
    device = torch.device("cpu")

    wCont = torch.zeros(x_train.shape[1], dtype=dtype, device=device, requires_grad=True)
    bCont = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)
    wBin = torch.zeros(x_train.shape[1], dtype=dtype, device=device, requires_grad=True)
    bBin = torch.zeros(1, dtype=dtype, device=device, requires_grad=True)

    lamdaCont_max = max(lamdaCont_path)
    lamdaBin_max = max(lamdaBin_path)

    lamdaCont_opt = lamdaCont_max
    lamdaBin_opt = lamdaBin_max

    wCont_dict = {}
    bCont_dict = {}
    errorCont_dict = {}
    wBin_dict = {}
    bBin_dict = {}
    errorBin_dict = {}

    for lamdaCont in lamdaCont_path:
        for lamdaBin in lamdaBin_path:
            print("lCont = ", lamdaCont, " lBin = ", lamdaBin)
            print("----------------------")

            result = ind_TwoVar_glmnet(x_train, yCont_train, yBin_train,
                                       alpha, lamdaCont, lamdaBin,
                                       wCont.clone().detach().requires_grad_(True),
                                       bCont.clone().detach().requires_grad_(True),
                                       wBin.clone().detach().requires_grad_(True),
                                       bBin.clone().detach().requires_grad_(True),
                                       max_iterations, learning_rate, verbose,
                                       threshold_error, p, q, areIndependent)

            wCont = result.wCont.clone().detach()
            bCont = result.bCont.clone().detach()
            wBin = result.wBin.clone().detach()
            bBin = result.bBin.clone().detach()

            lossCont = torch.sqrt(MSE(yCont_validation, torch.matmul(x_validation, wCont) + bCont)).item()
            lossBin = logLoss(yBin_validation, torch.sigmoid(bBin + torch.matmul(x_validation, wBin))).item()

            wCont_dict[(lamdaCont, lamdaBin)] = wCont
            bCont_dict[(lamdaCont, lamdaBin)] = bCont
            errorCont_dict[(lamdaCont, lamdaBin)] = lossCont

            wBin_dict[(lamdaCont, lamdaBin)] = wBin
            bBin_dict[(lamdaCont, lamdaBin)] = bBin
            errorBin_dict[(lamdaCont, lamdaBin)] = lossBin

            if lossCont + lossBin < errorCont_dict[(lamdaCont_opt, lamdaBin_opt)] + errorBin_dict[(lamdaCont_opt, lamdaBin_opt)]:
                lamdaCont_opt = lamdaCont
                lamdaBin_opt = lamdaBin

    class Result:
        def __init__(self, wCont_dict, bCont_dict, errorCont_dict, lamdaCont_opt, lamdaCont_max,
                     wBin_dict, bBin_dict, errorBin_dict, lamdaBin_opt, lamdaBin_max):
            self.wCont_dict = wCont_dict
            self.bCont_dict = bCont_dict
            self.errorCont_dict = errorCont_dict
            self.lamdaCont_opt = lamdaCont_opt
            self.lamdaCont_max = lamdaCont_max
            self.wBin_dict = wBin_dict
            self.bBin_dict = bBin_dict
            self.errorBin_dict = errorBin_dict
            self.lamdaBin_opt = lamdaBin_opt
            self.lamdaBin_max = lamdaBin_max
            self.wCont = wCont_dict[(lamdaCont_opt, lamdaBin_opt)]
            self.bCont = bCont_dict[(lamdaCont_opt, lamdaBin_opt)]
            self.wBin = wBin_dict[(lamdaCont_opt, lamdaBin_opt)]
            self.bBin = bBin_dict[(lamdaCont_opt, lamdaBin_opt)]

    return Result(wCont_dict, bCont_dict, errorCont_dict, lamdaCont_opt, lamdaCont_max,
                  wBin_dict, bBin_dict, errorBin_dict, lamdaBin_opt, lamdaBin_max)



def cv_TwoVar_glmnet(x, yCont, yBin,
                         alpha, threshold_error=1e-5, max_iterations=25,
                         learning_rate=0.01, lambdas=10, k=5, verbose=True,
                         p=1, q=1, areIndependent=False):
    # Perform cross-validation for glmnet with univariate feature selection
    start_time = time.time()
    validationCont_errors = {}
    validationBin_errors = {}

    kf = KFold(n_splits=k, shuffle=True)
    wCont_folds = {}
    bCont_folds = {}
    wBin_folds = {}
    bBin_folds = {}


    i = 0

    x_t = torch.transpose(x, 0, 1)

    lamdaCont_max = max([torch.matmul(x_t[i], yCont - yCont.mean()).abs() for i in range(len(x_t))]) / (len(x_t)*alpha)
    lamdaBin_max = max([torch.matmul(x_t[i], yBin - yBin.mean()).abs() for i in range(len(x_t))]) / (len(x_t)*alpha)


    lamdaCont_min = 0.001 * lamdaCont_max
    lamdaBin_min = 0.001 * lamdaBin_max

    pathCont = (np.power(10, np.linspace(np.log10(lamdaCont_min), np.log10(lamdaCont_max), lambdas)))[::-1]
    pathBin = (np.power(10, np.linspace(np.log10(lamdaBin_min), np.log10(lamdaBin_max), lambdas)))[::-1]

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        yCont_train, yCont_val = yCont[train_index], yCont[val_index]
        yBin_train, yBin_val = yBin[train_index], yBin[val_index]


        ret = std_TwoVar_glmnet(x_train, x_val, yCont_train, yCont_val, yBin_train, yBin_val,
                                alpha, pathCont, pathBin,
                                threshold_error=threshold_error,
                                max_iterations=max_iterations,
                                learning_rate=learning_rate,
                                verbose=verbose, p=p, q=q, areIndependent=areIndependent)

        wCont_folds[i] = ret.wCont_dict
        bCont_folds[i] = ret.bCont_dict
        wBin_folds[i] = ret.wBin_dict
        bBin_folds[i] = ret.bBin_dict
        validationCont_errors[i] = ret.errorCont_dict
        validationBin_errors[i] = ret.errorBin_dict

        i += 1

    class Result:
        def __init__(self, validationCont_errors, validationBin_errors,
                     pathCont, pathBin,
                     wCont_folds, bCont_folds, wBin_folds, bBin_folds,
                     lamdaCont_max, lamdaBin_max, start_time):

            self.validationCont_errors = validationCont_errors
            self.validationBin_errors = validationBin_errors
            self.pathCont = pathCont
            self.pathBin = pathBin
            self.wCont_folds = wCont_folds
            self.bCont_folds = bCont_folds
            self.wBin_folds = wBin_folds
            self.bBin_folds = bBin_folds
            self.lamdaCont_max = lamdaCont_max
            self.lamdaBin_max = lamdaBin_max
            self.start_time = start_time

            self.wCont = {}
            self.bCont = {}
            self.wBin = {}
            self.bBin = {}
            self.validationCont_errors_means = {}
            self.validationCont_errors_sd = {}
            self.validationBin_errors_means = {}
            self.validationBin_errors_sd = {}


            for key in wCont_folds[0].keys():
                self.wCont[key] = torch.zeros(len(x_t), dtype=torch.float, device=torch.device("cpu"))
                self.bCont[key] = torch.zeros(1, dtype=torch.float, device=torch.device("cpu"))
                for i in range(k):
                    self.wCont[key] += wCont_folds[i][key]
                    self.bCont[key] += bCont_folds[i][key]
                self.wCont[key] = self.wCont[key] / k
                self.bCont[key] = self.bCont[key] / k

            for key in wBin_folds[0].keys():
                self.wBin[key] = torch.zeros(len(x_t), dtype=torch.float, device=torch.device("cpu"))
                self.bBin[key] = torch.zeros(1, dtype=torch.float, device=torch.device("cpu"))
                for i in range(k):
                    self.wBin[key] += wBin_folds[i][key]
                    self.bBin[key] += bBin_folds[i][key]
                self.wBin[key] = self.wBin[key] / k
                self.bBin[key] = self.bBin[key] / k

            for key in validationCont_errors[0].keys():
                self.validationCont_errors_means[key] = 0
                self.validationCont_errors_sd[key] = 0
                error_list = []
                for i in range(k):
                    self.validationCont_errors_means[key] += validationCont_errors[i][key]
                    error_list.append(validationCont_errors[i][key])
                self.validationCont_errors_means[key] = self.validationCont_errors_means[key] / k
                self.validationCont_errors_sd[key] = np.std(error_list) / np.sqrt(len(pathCont))

            for key in validationBin_errors[0].keys():
                self.validationBin_errors_means[key] = 0
                self.validationBin_errors_sd[key] = 0
                error_list = []
                for i in range(k):
                    self.validationBin_errors_means[key] += validationBin_errors[i][key]
                    error_list.append(validationBin_errors[i][key])
                self.validationBin_errors_means[key] = self.validationBin_errors_means[key] / k
                self.validationBin_errors_sd[key] = np.std(error_list) / np.sqrt(len(pathBin))

            self.lambdaCont_opt = min(self.validationCont_errors_means, key=self.validationCont_errors_means.get)
            self.lambdaBin_opt = min(self.validationBin_errors_means, key=self.validationBin_errors_means.get)

            self.wCont_opt = self.wCont[self.lambdaCont_opt]
            self.bCont_opt = self.bCont[self.lambdaCont_opt]
            self.wBin_opt = self.wBin[self.lambdaBin_opt]
            self.bBin_opt = self.bBin[self.lambdaBin_opt]

            intervaloCont = (self.validationCont_errors_means[self.lambdaCont_opt] - self.validationCont_errors_sd[self.lambdaCont_opt],
                             self.validationCont_errors_means[self.lambdaCont_opt] + self.validationCont_errors_sd[self.lambdaCont_opt])
            intervaloBin = (self.validationBin_errors_means[self.lambdaBin_opt] - self.validationBin_errors_sd[self.lambdaBin_opt],
                            self.validationBin_errors_means[self.lambdaBin_opt] + self.validationBin_errors_sd[self.lambdaBin_opt])

            self.lambdaCont_1se = max([k for k in self.validationCont_errors_means.keys()
                                       if intervaloCont[0] <= self.validationCont_errors_means[k] <= intervaloCont[1]],
                                      key=self.validationCont_errors_means.get)
            self.lambdaBin_1se = max([k for k in self.validationBin_errors_means.keys()
                                      if intervaloBin[0] <= self.validationBin_errors_means[k] <= intervaloBin[1]],
                                     key=self.validationBin_errors_means.get)

            self.wCont_1se = self.wCont[self.lambdaCont_1se]
            self.bCont_1se = self.bCont[self.lambdaCont_1se]
            self.wBin_1se = self.wBin[self.lambdaBin_1se]
            self.bBin_1se = self.bBin[self.lambdaBin_1se]

            def predict_opt(x):
                return (torch.matmul(x, self.wCont_opt) + self.bCont_opt,
                        classify(torch.sigmoid(self.bBin_opt + torch.matmul(x, self.wBin_opt))))

            def predict_1se(x):
                return (torch.matmul(x, self.wCont_1se) + self.bCont_1se,
                        classify(torch.sigmoid(self.bBin_1se + torch.matmul(x, self.wBin_1se))))


            self.predict_opt = predict_opt
            self.predict_1se = predict_1se

            self.validationCont_errors_means_low = [self.validationCont_errors_means[key] - 1.96*self.validationCont_errors_sd[key]
                                                    for key in self.validationCont_errors_means.keys()]
            self.validationCont_errors_means_high = [self.validationCont_errors_means[key] + 1.96*self.validationCont_errors_sd[key]
                                                     for key in self.validationCont_errors_means.keys()]
            self.validationBin_errors_means_low = [self.validationBin_errors_means[key] - 1.96*self.validationBin_errors_sd[key]
                                                   for key in self.validationBin_errors_means.keys()]
            self.validationBin_errors_means_high = [self.validationBin_errors_means[key] + 1.96*self.validationBin_errors_sd[key]
                                                    for key in self.validationBin_errors_means.keys()]


            def error_evolution():
                plt.figure()
                plt.title('Continuous validation error evolution')
                plt.fill_between(pathCont, self.validationCont_errors_means_low, self.validationCont_errors_means_high,
                                 color='blue', alpha=0.2)
                plt.semilogx(pathCont, self.validationCont_errors_means.values(), lw=1, color='blue', label='Error')
                plt.axvline(x=self.lambdaCont_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambdaCont_1se, color='green', label='lambda_1se')

                plt.xticks([min((pathCont)), (self.lambdaCont_opt), (self.lambdaCont_1se), max((pathCont))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend()
                plt.show()

                plt.figure()
                plt.title('Binary validation error evolution')
                plt.fill_between(pathBin, self.validationBin_errors_means_low, self.validationBin_errors_means_high,
                                 color='blue', alpha=0.2)
                plt.semilogx(pathBin, self.validationBin_errors_means.values(), lw=1, color='blue', label='Error')
                plt.axvline(x=self.lambdaBin_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambdaBin_1se, color='green', label='lambda_1se')

                plt.xticks([min((pathBin)), (self.lambdaBin_opt), (self.lambdaBin_1se), max((pathBin))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend()
                plt.show()



            def coefficients_evolution():
                plt.figure()
                plt.title('Continuous coefficient evolution')

                transposed_dict = {}

                for key, value in self.wCont.items():
                    for i, v in enumerate(value):
                        transposed_dict[i] = transposed_dict.get(i, []) + [v.item()]

                for key in transposed_dict.keys():
                    plt.semilogx(pathCont, transposed_dict[key], lw=1, label='wCont_' + str(key))

                plt.axvline(x=self.lambdaCont_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambdaCont_1se, color='green', label='lambda_1se')

                plt.xticks([min((pathCont)), (self.lambdaCont_opt), (self.lambdaCont_1se), max((pathCont))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend()
                plt.show()

                plt.figure()
                plt.title('Binary coefficient evolution')

                transposed_dict = {}

                for key, value in self.wBin.items():
                    for i, v in enumerate(value):
                        transposed_dict[i] = transposed_dict.get(i, []) + [v.item()]

                for key in transposed_dict.keys():
                    plt.semilogx(pathBin, transposed_dict[key], lw=1, label='wBin_' + str(key))

                plt.axvline(x=self.lambdaBin_opt, color='red', label='lambda_opt')
                plt.axvline(x=self.lambdaBin_1se, color='green', label='lambda_1se')

                plt.xticks([min((pathBin)), (self.lambdaBin_opt), (self.lambdaBin_1se), max((pathBin))],
                           [r'$\lambda_{min}$', r'$\lambda_{opt}$', r'$\lambda_{1se}$', r'$\lambda_{max}$'])
                plt.legend()
                plt.show()


            self.error_evolution = error_evolution
            self.coefficients_evolution = coefficients_evolution
            self.time = time.time() - start_time

    return Result(validationCont_errors, validationBin_errors, pathCont, pathBin,
                  wCont_folds, bCont_folds,wBin_folds, bBin_folds, lamdaCont_max, lamdaBin_max,
                  start_time)
