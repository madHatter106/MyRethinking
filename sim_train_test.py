import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as pl
%matplotlib inline

N = 20
k = 3
rho = np.array([.15, -0.4])
b_sigma = 100

n_dim = 1 + rho.size
if n_dim < k: ndim = k
Rho = np.diagflat(np.ones(n_dim))
Rho[0, 1: rho.size+1] = rho
Rho = Rho.T + np.triu(Rho,1)
X_train = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=Rho, size=N)
X_test = np.random.multivariate_normal(mean=np.zeros(n_dim), cov=Rho, size=N)
y_train = X_train[:, 0]
mm_train = np.ones((N, 1))

if k > 1:
    mm_train = np.c_[mm_train, X_train[:, 1:]]
    bnames = ["b%d" % i for i in range(1, k)]
bnames = ["a"] + bnames
bnames
mm_train

d = pd.DataFrame(np.c_[y_train, mm_train], columns=)
