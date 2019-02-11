import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as ar
from sklearn.preprocessing import scale
import matplotlib.pyplot as pl
from causalgraphicalmodels import CausalGraphicalModel

d = pd.read_csv('rethinking-Experimental/data/milk.csv', delimiter=';')

d['K'] = scale(d['kcal.per.g'].values)
d['N'] = scale(d['neocortex.perc'].values)
d['M'] = scale(d['mass'].values)

dcc = d.dropna()
"""
Model with very vague prior
"""
with pm.Model() as m5_5_draft:
    a = pm.Normal('a', 0, 1)
    bN = pm.Normal('bN', 0, 1)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * dcc.N
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=dcc.K)
priors = pm.sample_prior_predictive(model=m5_5_draft, vars=['a','bN'])
x_seq = np.linspace(-2, 2).reshape(1,-1)
a_5d = priors['a'].reshape(-1, 1)
bN_5d = priors['bN'].reshape(-1, 1)
mu_5d = a_5d + bN_5d * x_seq

with m5_5_draft:
    trace_5_5_draft = pm.sample()

with pm.Model() as m5_5:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    sigma = pm.Exponential('sigma', 1)
    mu = a + bN * dcc.N
    Ki = pm.Normal('Ki', mu=mu, sd=sigma, observed=dcc.K)

priors_5 = pm.sample_prior_predictive(model=m5_5, vars=['a', 'bN'])
a_5 = priors_5['a'].reshape(-1, 1)
bN_5 = priors_5['bN'].reshape(-1, 1)
mu_5 = a_5 + bN_5 * x_seq

_, ax = pl.subplots(ncols=2, figsize=(10, 4))
ax[0].plot(x_seq.T, mu_5d.T, color='k', alpha=0.1)
ax[1].plot(x_seq.T, mu_5.T, color='k', alpha=0.1)
ax[0].set_ylim(-2, 2)
ax[1].set_ylim(-2, 2);
ax[0].set_title('vague bN prior')
ax[1].set_title('weakly informative bN prior');

with m5_5:
    trace_5 = pm.sample(tune=1000)
pm.summary(trace_5, alpha=0.11)

a5_post = trace_5.get_values(varname='a').reshape(-1, 1)
bN5_post = trace_5.get_values(varname='bN').reshape(-1, 1)
muN_post = a5_post + bN5_post*x_seq
hpd_N = pm.hpd(muN_post)

"""
Using mass M as predictor for K
"""
with pm.Model() as m5_6:
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    sigma = pm.Exponential('sigma', 1)
    mu = a + bM * dcc.M
    Ki = pm.Normal('Ki', mu=mu, sd=sigma, observed=dcc.K)
    trace_6 = pm.sample(tune=1000)


pm.summary(trace_6, alpha=0.11)

a6_post = trace_6.get_values(varname='a').reshape(-1, 1)
bM6_post = trace_6.get_values(varname='bM').reshape(-1, 1)
muM_post = a6_post + bM6_post * x_seq
hpd_M = pm.hpd(muM_post)

_, ax = pl.subplots(ncols=2, figsize=(10, 4),)
ax[0].plot(x_seq.T, muN_post.mean(axis=0), color='k')
ax[0].fill_between(x_seq.flatten(), hpd_N[:, 0], hpd_N[:, 1], color='k', alpha=0.5)
ax[1].plot(x_seq.T, muM_post.mean(axis=0), color='k')
ax[1].fill_between(x_seq.flatten(), hpd_M[:, 0], hpd_M[:, 1], color='k', alpha=0.5)
dcc.plot(x='N', y='K', kind='scatter', ax=ax[0])
dcc.plot(x='M', y='K', kind='scatter', ax=ax[1]);

"""
Next is to make a multivariate model and work some counterfactuals in an attempt to discern a
causal link.
"""

with pm.Model() as m5_7:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    sigma = pm.Exponential('sigma', 1)
    mu = a + bN * dcc.N + bM * dcc.M
    Ki = pm.Normal('Ki', mu=mu, sd=sigma, observed=dcc.K)
    trace_7 = pm.sample(tune=1000)
pm.summary(trace_7, alpha=0.11)

pm.forestplot([trace_7, trace_6, trace_5], models=['m7', 'm6', 'm5'],
             varnames=['bM', 'bN'], rhat=False, alpha=0.11);
pd.plotting.scatter_matrix(dcc[['M', 'N', 'K']]);

"""
The regression model (m5_7) asks if high N is associated with high K. Likewise m5_7 asks whether
high M implies high K. Bigger species like apes (big M) have milk with less energy. But spp with
more neocortex (big N) have richer milk (big K). The fact that M and N are correlated makes these
relationships difficult to see unless both factors are accounted for.
----o----

Simulating a Masking Relationship. Two predictors (M, N) are correlated with one another, and one (M)
is positively correlated with the target (K) while the other (N) is negatively correlated with K
"""
div_msk = CausalGraphicalModel(nodes=['M', 'N', 'K'],
                               edges=[('M', 'K'), ('N', 'K'), ('M', 'N')])
div_msk.draw()

n = 100
M = np.random.normal(size=n)
N = np.random.normal(loc=M, size=n)
K = np.random.normal(loc=N-M, size=n)

d_sim = pd.DataFrame(dict(K=K, M=M, N=N))
pd.plotting.scatter_matrix(d_sim, alpha=0.5, diagonal=);


with pm.Model() as m5_sim:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim.N
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim.K)
    tr5s = pm.sample()

with pm.Model() as m6_sim:
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bM * d_sim.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim.K)
    tr6s = pm.sample()

with pm.Model() as m7_sim:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim.N + bM * d_sim.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim.K)
    tr7s = pm.sample()


ar.plot_forest([tr7s, tr6s, tr5s], model_names=['m7', 'm6', 'm5'], var_names=['bM', 'bN'],
               combined=True, figsize=(5, 5), );
pl.axvline(ls='--', color='k', zorder=0);
"""
What about the following relationship, where the predictor causal flow is reversed:
"""
div_msk2 = CausalGraphicalModel(nodes=['K', 'M', 'N'],
                                edges=[('N', 'M'),('M', 'K'), ('N', 'K')])
div_msk2.draw()

N = np.random.normal(size=n)
M = np.random.normal(N)
K = np.random.normal(N-M)

d_sim2 = pd.DataFrame(dict(K=K, M=M, N=N ))
pd.plotting.scatter_matrix(d_sim2, diagonal='kde');

with pm.Model() as m5_sim2:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim2.N
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim2.K)
    tr5s2 = pm.sample()

with pm.Model() as m6_sim2:
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bM * d_sim2.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim2.K)
    tr6s2 = pm.sample()

with pm.Model() as m7_sim2:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim2.N + bM * d_sim2.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim2.K)
    tr7s2 = pm.sample()
    ar.plot_forest([tr7s2, tr6s2, tr5s2], model_names=['m7', 'm6', 'm5'], var_names=['bM', 'bN'],
                   combined=True, figsize=(5, 5), );
    pl.axvline(ls='--', color='k', zorder=0);


"""
Third possibility, an undetected variable, U, causal parent to both M and N:
"""

div_msk3 = CausalGraphicalModel(nodes=['U', 'M', 'N', 'K'],
                                edges=[('U', 'M'), ('U', 'N'),
                                       ('M', 'K'), ('N', 'K')])
div_msk3.draw()

U = np.random.normal(size=n)
M = np.random.normal(U)
N = np.random.normal(U)
K = np.random.normal(N-M)

d_sim3 = pd.DataFrame(dict(K=K, M=M, N=N, U=U))
pd.plotting.scatter_matrix(d_sim3, diagonal='kde', edgecolor='k');

with pm.Model() as m5_sim3:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim3.N
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim3.K)
    tr5s3 = pm.sample()

with pm.Model() as m6_sim3:
    a = pm.Normal('a', 0, 0.2)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bM * d_sim3.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim3.K)
    tr6s3 = pm.sample()

with pm.Model() as m7_sim3:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim3.N + bM * d_sim3.M
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim3.K)
    tr7s3 = pm.sample()


ar.plot_forest([tr7s3, tr6s3, tr5s3], model_names=['m7', 'm6', 'm5'], var_names=['bM', 'bN'],
               combined=True, figsize=(5, 5), );
pl.axvline(ls='--', color='k', zorder=0);


"""adding a model for the scenario where we observe U"""
with pm.Model() as m8_sim3:
    a = pm.Normal('a', 0, 0.2)
    bN = pm.Normal('bN', 0, 0.5)
    bM = pm.Normal('bM', 0, 0.5)
    bU = pm.Normal('bU', 0, 0.5)
    σ = pm.Exponential('sigma', 1)
    mu = a + bN * d_sim3.N + bM * d_sim3.M + bU * d_sim3.U
    Ki = pm.Normal('Ki', mu=mu, sd=σ, observed=d_sim3.K)
    tr8s3 = pm.sample(tune=1000)


ar.plot_forest([tr8s3, tr7s3, tr6s3, tr5s3], model_names=['m8', 'm7', 'm6', 'm5'], var_names=['bU','bM', 'bN'],
               combined=True, figsize=(5, 5), );
pl.axvline(ls='--', color='k', zorder=0);
