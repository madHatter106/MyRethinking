import numpy as np
import pandas as pd
import pymc3 as pm
from theano import shared

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as pl
#%matplotlib inline




def simulate(N=100):
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
    dm = dict.fromkeys(['m%d' % i for i in range(1, 6)])
    for i in range(1, 6):
        x_shared = shared(X_train[:,:i])
        y_shared = shared(y_train)
        with pm.Model() as m:
            β = pm.Normal('β', 0, 1, shape=i)
            μ = pm.math.dot(x_shared, β)
            σ = pm.Exponential('σ', 1)
            y = pm.Normal('y', mu=μ, sd=σ, observed=y_shared)
            trace = pm.sample(200, tune=1000)
        waic_train = pm.waic(trace=trace, model=m)
        deviance_train = waic_train.WAIC - 2 * waic_train.p_WAIC
        x_shared.set_value(X_test[:, :i])
        y_shared.set_value(y_test)
        waic_test = pm.waic(trace=trace, model=m)
        deviance_test = waic_test.WAIC - 2 * waic_test.p_WAIC
        return deviance_train, deviance_test
