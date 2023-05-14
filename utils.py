import cvxopt as cv
import cvxpy as cp
import numpy as np

def is_integer_solution(solution: list, epsilon: float):
    return all([abs(x - int(x)) <= epsilon for x in solution])
