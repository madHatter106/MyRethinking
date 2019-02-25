import numpy as np
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel
import arviz as ar
import pymc3 as pm
dag_6_1 = CausalGraphicalModel(nodes=['X', 'Y', 'U',
                                      'A', 'B', 'C'],
                               edges=[('X', 'Y'),
                                      ('C', 'Y'),
                                      ('U', 'X'),
                                      ('U', 'B'),
                                      ('A', 'U'),
                                      ('A', 'C'),
                                      ('C', 'B')])
dag_6_1.get_all_backdoor_adjustment_sets(x='X', y='Y', )
dag_6_1.draw()
"""
Waffle house example, where:
S: whether a State is in the Southern US
A: median age at marriage
M: marriage multivariate
W: number of Waffle Houses
D: divorce rate.

Assumptions:
------------
Southern States have lower age of marriage: (S->A)
S have higher rates of mariage, directly (S->M)
    and mediated through age of marriage (S->A->M)
S have more waffles (S->W)
Also, A->D and M->D.

"""
dag_6_2 = CausalGraphicalModel(nodes=['A', 'M', 'D',
                                      'W', 'S'],
                               edges=[('S', 'M'),
                                      ('S', 'W'),
                                      ('S', 'A'),
                                      ('A', 'M'),
                                      ('A', 'D'),
                                      ('W', 'D'),
                                      ('M', 'D')])

dag_6_2.draw()
dag_6_2.get_all_backdoor_adjustment_sets('W', 'D')

dag_6_2.get_all_independence_relationships()

d = pd.read_csv('./rethinking-Experimental/data/WaffleDivorce.csv',
                delimiter=';')
d.head(2).T
d_ = d[['MedianAgeMarriage', 'Marriage',
        'WaffleHouses', 'South', 'Divorce']]
A = (d_.MedianAgeMarriage - d_.MedianAgeMarriage.mean()
     ) / (2 * d_.MedianAgeMarriage.std())
M = (d_.Marriage - d_.Marriage.mean()
     ) / (2 * d_.MedianAgeMarriage.std())
W = (d_.WaffleHouses - d_.WaffleHouses.mean()
     ) / (2 * d_.WaffleHouses.std())
S = d_.South
D = (d_.Divorce - d_.Divorce.mean()
     ) / (2 * d_.Divorce.std())
d_s = pd.DataFrame(dict(A=A.values, M=M.values,
                        W=W.values, S=S.values,
                        D=D.values))
d_s.describe()
with pm.Model() as waf1:
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', 0, 1)
    β_W = pm.Normal('β_W', 0, 1)
    μ = α + β_W * d_s.W.values
    D = pm.Normal('D', mu=μ, sd=σ,
                  observed=d_s.D.values)

pm.model_to_graphviz(waf1)
with waf1:
    trc_waf1 = pm.sample(tune=1000)
pm.summary(trc_waf1, alpha=0.11)
ar.plot_posterior(trc_waf1, round_to=2,
                  credible_interval=0.89);

with pm.Model() as waf2:
    σ = pm.Exponential('σ', 1)
    α = pm.Normal('α', 0, 1)
    β_W = pm.Normal('β_W', 0, 1)
    β_S = pm.Normal('β_S', 0, 1)
    μ = α + β_W * d_s.W.values + β_S * d_s.S.values
    D = pm.Normal('D', mu=μ, sd=σ, observed=d_s.D.values)

with waf2:
    trc_waf2 = pm.sample(tune=1000)
pm.summary(trc_waf2, alpha=0.11)
ar.plot_posterior(trc_waf2, round_to=2, kind='hist',
                  credible_interval=.89);
