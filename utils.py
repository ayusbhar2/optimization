import cvxopt as cv
import cvxpy as cp
import numpy as np

def get_lp_relaxation_of_bip(bip):
    new_constraints = []
    for v in bip.variables():
        if v.attributes['boolean']:
            v.attributes['boolean'] = False
            new_constraints += [0 <= v, v <= 1]
    lp = cp.Problem(bip.objective, bip.constraints + new_constraints)
    return lp
