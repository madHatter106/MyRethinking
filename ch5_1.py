import numpy as np
import pandas as pd
import pymc3 as pm
from seaborn import kdeplot
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel

d = pd.read_csv('rethinking-Experimental/data/WaffleDivorce.csv',
              delimiter=';')
d['A'] = scale(d.MedianAgeMarriage.values, with_std=False)
d['D'] = scale(d.Divorce.values, with_std=False)
with pm.Model() as m5_1:
    a = pm.Normal('a', 0, 0.2)
    #a = pm.Normal('a' ,0 ,1)
    bA = pm.Normal('bA', 0, 0.5)
    #bA = pm.Normal('bA', 0, 1)
    #bA = pm.HalfNormal('bA', 1)
    mu = a + bA * d.A
    sigma = pm.Exponential('sigma', lam=1)
    D = pm.Normal('D', mu, sigma, observed=d.D)

np.random.seed(10)
prior = pm.sample_prior_predictive(model=m5_1)
prior.keys()
A = np.linspace(-2, 2).reshape(-1, 1)
prior_a = prior['a'].reshape(1, -1)
prior_bA = prior['bA'].reshape(1, -1)
D = prior_a + prior_bA * A
pl.plot(A, D, color='k', alpha=0.1);

pl.plot(A, D, color='k', alpha=0.1);

pl.plot(A, D, color='k', alpha=0.1);


d['M'] = scale(d.Marriage.values, with_std=False)

with pm.Model() as m5_2:
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    mu = a + bM * d.M
    sigma = pm.Exponential('sigma', lam=1)
    D = pm.Normal('D', mu, sigma, observed=d.D)

div_dag = CausalGraphicalModel(nodes=['A', 'M', 'D'],
                               edges=[('A', 'M'),
                                      ('A', 'D'),
                                      ('M', 'D')])
div_dag2 = CausalGraphicalModel(nodes=['A', 'M', 'D'],
                                edges=[('A', 'M'),
                                       ('A', 'D')])

with m5_1:
    trace_5_1 = pm.sample(1000, tune=1000)
with m5_2:
    trace_5_2 = pm.sample(1000, tune=1000)
pm.forestplot(trace_5_1, varnames=['a', 'bA', 'sigma']);
pm.forestplot(trace_5_2, varnames=['a', 'bM', 'sigma'])
# The above is consistent with two causal DAGS:
div_dag.draw()
# and
div_dag2.draw()
"""
We need a model that CONTROLS FOR A while assessing the association between M and D
Fit a multiple regression to predict divorce using both marriage rate (M) and age at
marriage (A) to answer the questions:

1. Knowing marriage rate (M), what additional value is there in knowing age at marriage (A)
2. Knowing age at marriage (A), what additional value is there in knowing marriage rate (M)
"""

with pm.Model() as m5_3:
    sigma = pm.Exponential('sigma', lam=1)
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    bA = pm.Normal('bA', 0, 0.5)
    mu_i = a + bM * d.M + bA * d.A
    Di = pm.Normal('Di', mu=mu_i, sd=sigma, observed=d.D)

with m5_3:
    trace_5_3 = pm.sample(1000, tune=1000)
pm.forestplot([trace_5_1, trace_5_2, trace_5_3], models=['m5_1', 'm5_2', 'm5_3'],
              varnames=['bA', 'bM'], rhat=False);
"""
With m5_3: bA doesn't move, just grows a bit more uncertain and bM is only associated with D when
age at marriage (A) is missing, implying there is no/almost no direct causal link between marriage
rate (M) and divorce (D); the association is spurious. For a complete picture, model the association
between M and A:
"""
with pm.Model() as m5_3b:
    sigma = pm.Exponential('sigma', lam=1)
    a = pm.Normal('a', 0, 0.2)
    bA_M = pm.Normal('bA_M',0, 0.5)
    mu_i = a + bA_M * d.A
    Mi = pm.Normal('Mi', mu_i, sigma, observed=d.M)
    trace_5_3_b = pm.sample(1000, tune=1000)
pm.forestplot(trace_5_3_b, varnames=['bA_M', 'a'], rhat=False);

"""
Simulating the divorce example: Here I simulate the causal relationship M <-- A --> D and rerun
the models above

"""
div_dag2.draw()
N = 50
d_ = pd.DataFrame(columns=['age', 'mar', 'div'])
d_['age'] = np.random.randn(N)          # sim A
d_['mar'] = np.random.randn(N) + d_.age # sim A --> M
d_['div'] = np.random.randn(N) + d_.age
d_.head()


with pm.Model() as m5_1s:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bA = pm.Normal('bA', 0, 0.5)
    mu_i = a + bA * d_.age
    di = pm.Normal('di', mu_i, sd=sig, observed=d_['div'])
    trc_5_1s = pm.sample()

with pm.Model() as m5_2s:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    mu = a + bM * d_.mar
    di = pm.Normal('di', mu, sd=sig, observed=d_['div'])
    trc_5_2s = pm.sample()

with pm.Model() as m5_3s:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bA = pm.Normal('bA', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    mu = a + bA * d_.age + bM * d_.mar
    di = pm.Normal('di', mu, sd=sig, observed=d_['div'])
    trc_5_3s = pm.sample()

pm.forestplot([trc_5_1s, trc_5_2s, trc_5_3s], models=['m5_1s', 'm5_2s', 'm5_3s'],
              varnames=['bA', 'bM'], rhat=False);

"""
Simulating divorce example with A and M influence on D:
"""
div_dag.draw()
d_['div2'] = np.random.randn(N) + d_.age + d_.mar

with pm.Model() as m5_1s2:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bA = pm.Normal('bA', 0, 0.5)
    mu_i = a + bA * d_.age
    di = pm.Normal('di', mu_i, sd=sig, observed=d_['div2'])
    trc_5_1s2 = pm.sample()

with pm.Model() as m5_2s2:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    mu = a + bM * d_.mar
    di = pm.Normal('di', mu, sd=sig, observed=d_['div2'])
    trc_5_2s2 = pm.sample()

with pm.Model() as m5_3s2:
    sig = pm.Exponential('sig', lam=1)
    a = pm.Normal('a' ,0, 0.2)
    bA = pm.Normal('bA', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    mu = a + bA * d_.age + bM * d_.mar
    di = pm.Normal('di', mu, sd=sig, observed=d_['div2'])
    trc_5_3s2 = pm.sample()

pm.forestplot([trc_5_1s2, trc_5_2s2, trc_5_3s2], models=['m5_1s2', 'm5_2s2', 'm5_3s2'],
              varnames=['bA', 'bM'], rhat=False);
