import cvxpy as cp
import pandas as pd


def is_integer_solution(solution: list, epsilon: float):
    return all([abs(x - int(x)) <= epsilon for x in solution])


def get_result_summary(prob: cp.problems.problem.Problem):
    result = {'status': None,
              'optimal_value': None,
              'optimal_solution': None}

    result['status'] = prob.status
    result['optimal_value'] = prob.value
    if prob.status == 'optimal':
        result['optimal_solution'] = dict(
            [(v.name(), v.value[0]) for v in prob.variables()])

    return result


def prettify(d: dict):
    variable = []
    value = []
    for k, v in sorted(d.items()):
        variable.append(k)
        value.append(v)
    df = pd.DataFrame({'variable': variable, 'value': value})
    return df


def check_violation(constraints: list):
    for c in constraints:
        if not c.value():
            return c
    return None
