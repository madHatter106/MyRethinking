import numpy as np
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel

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
