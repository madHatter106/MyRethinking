import numpy as np
import pandas as pd
import pymc3 as pm
from theano import shared

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as pl
%matplotlib inline

"""Simulate function development below"""
N = 100
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
d = pd.DataFrame(np.c_[y_train, X_train],
                 columns=['y'] + ['x%d' %i for i in range(X_train.shape[1])])
d.head()

pd.plotting.scatter_matrix(d[['y'] + ['x%d' %i for i in range(1, 5)]]);

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
    loo_train = pm.loo(trace=trace, model=m)
    x_shared.set_value(X_test[:, :i])
    y_shared.set_value(y_test)
    waic_test = pm.waic(trace=trace, model=m)
    loo_test = pm.loo(trace=trace, model=m)
    dm['m%d' %i] = dict(model=m, trace=trace,
                        waic_train=waic_train, waic_test=waic_test,
                        loo_train=loo_train, loo_test=loo_test)

deviance = dict.fromkeys(dm.keys())
for key in deviance.keys():
    deviance[key] = dict(train=(dm[key]['waic_train'].WAIC) -2 * dm[key]['waic_train'].p_WAIC,
                         test=(dm[key]['waic_test'].WAIC ) -2 * dm[key]['waic_test'].p_WAIC)
waic_train = [dm['m%d' %i]['waic_train'].WAIC for i in range(1, 6)]
test = [deviance['m%d' %i]['test'] for i in range(1, 6)]
train =[deviance['m%d' %i]['train']  for i in range(1, 6)]


test_r2 = []
train_r2 = []
test_mae = []
train_mae = []
for i in range(1, 6):
    β_ = dm[f'm{i}']['trace'].get_values('β').mean(axis=0)
    x_train = X_train[:, :i]
    x_test = X_test[:, :i]
    y_train_pred = x_train.dot(β_)
    y_test_pred = x_test.dot(β_)
    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))
train_payload = dict(marker='.', color='b', label='train')
test_payload = dict(marker='.', color='k', label='test')
f, ax = pl.subplots(nrows=3, figsize=(6, 8), sharex=True)
ax[0].plot(np.arange(1, 6), train_r2, **train_payload)#marker='.', color='b', label='train')
ax[0].plot(np.arange(1, 6), test_r2, **test_payload)
ax[0].set_ylabel(r'$R^{2}$')
ax[0].legend()
ax[1].plot(np.arange(1, 6), train_mae, **train_payload)
ax[1].plot(np.arange(1, 6), test_mae, **test_payload)
ax[1].set_ylabel(r'$MAE$')
ax[1].legend()
ax[2].plot(np.arange(1, 6).astype('int'), train, **train_payload)
ax[2].plot(np.arange(1, 6).astype('int'), test, **test_payload)
ax[2].plot(np.arange(1, 6).astype('int'), waic_train, marker='.', color='r', label='WAIC_train')
ax[2].legend()
ax[2].set_xticks(np.arange(1, 6))
ax[2].set_ylabel('deviance')
ax[2].set_xlabel('# parameters');
