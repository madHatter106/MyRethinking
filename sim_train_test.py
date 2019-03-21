import warnings
import numpy as np
import pandas as pd
import pymc3 as pm
from theano import shared
import scipy.stats as st

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae_score

import matplotlib.pyplot as pl
#%matplotlib inline

warnings.filterwarnings('ignore')

def simulate(N=100, n_params=5):
    n_dim = 5
    rho = np.array([.15, -0.4])
    b_sigma = 100

    Rho = np.diagflat(np.ones(n_dim))
    Rho[0, 1: rho.size+1] = rho
    Rho = Rho.T + np.triu(Rho,1)
    X_train = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=Rho, size=N)
    X_test = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=Rho, size=N)
    y_train = X_train[:, 0]
    y_test = X_test[:, 0]
    X_train = np.c_[np.ones((N, 1)), X_train[:, 1:]]
    X_test = np.c_[np.ones((N, 1)), X_test[:, 1:]]
    dm = dict.fromkeys(['m%d' % i for i in range(1, n_params)])
    deviance_train = np.empty(shape=n_params)
    r2_train = np.empty_like(deviance_train)
    mae_train = np.empty_like(deviance_train)
    deviance_test = np.empty_like(deviance_train)
    r2_test = np.empty_like(deviance_train)
    mae_test = np.empty_like(deviance_test)

    for i, n_param in enumerate(range(1, n_params+1)):
        X_ = X_train[:,:n_param]
        with pm.Model() as m:
            β = pm.Normal('β', 0, 1, shape=n_param)
            μ = pm.math.dot(X_, β)
            σ = pm.Exponential('σ', 1)
            y = pm.Normal('y', mu=μ, sd=σ, observed=y_train)
            theta = pm.find_MAP(progressbar=False)
        μ_train = np.dot(X_, theta['β'])
        μ_test = np.dot(X_test[:, :n_param], theta['β'])
        log_lkl_train = st.norm.logpdf(y_train, loc=μ_train, scale=theta['σ'])
        log_lkl_test = st.norm.logpdf(y_test, loc=μ_test, scale=theta['σ'])
        deviance_train[i] = -2 * log_lkl_train.sum()
        r2_train[i] = r2_score(y_train, μ_train)
        mae_train[i] = mae_score(y_train, μ_train)
        deviance_test[i] = -2 * log_lkl_test.sum()
        r2_test[i] = r2_score(y_test, μ_μ_test)
        mae_test[i] = mae_score(y_test, μ_test)

        train_dict = dict(r2=r2_train, mae=mae_train, deviance=deviance_train)
        test_dict = dict(r2=r2_test, mae=mae_test, deviance=deviance_test)

    return train_dict, test_dict
