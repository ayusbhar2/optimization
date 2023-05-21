# Introduction
The Unit Commitment Problem (UCP) [1] in power systems is the problem of scheduling the optimal number of power generating units (or simply units) to meet the total demand in each time period (or slot) of a planning horizon. The UCP is an optimization problem that aims to minmize the running costs, startup costs and shutdown costs of the units while satisfying constraints that guarantee appropriate operating conditions for the units.

The UCP can be formulated in a number of ways:
- combinatorial approach
- priority list / heuristic approach
- dynamic programming
- Lagrangian relaxation
- **Mixed Integer Programming (MIP)**

In this post we will consider the MIP approach since it is the state of the art for the UCP. Other approaches do not scale to any reasonable sized problem. We will consider a small but salient example [2] which has most features of a real life UCP.

# Example
We are given three units -  A, B and C - with their corresponding costs and operational constraints. The hourly demand  profile over a three-period planning horizon is also given. We want to find the optimal schedule that minimizes the total cost, meets the hourly demand requirements and satisfies the operational constraints.

|Unit|P_min (MW)|P_max (MW)|Min up (hr)|Min down (hr)|No-load cost (\$)|Marginal cost (\$/MWh)|Startup cost (\$)|Initial status|
|---|---| ---| ---|   ---|   ---|   ---|  ---|   ---|
|A   | 150| 250| 3|   3|   0|   10|  1000|    ON|
|B   | 50| 100| 2|   1|   0|   12|  600| OFF|
|C   |10  |50  |1   |1   |0   |20  |100 |OFF|

|Period| Demand (MW)|
|---|---|
|1 | 150|
|2 | 300|
|3 | 200|

We must call out the features of a real-life UCP that our example does NOT capture.
- Ignores unit rampup constraints
- Ignores reserve constraints and costs of reserves.
- Ignores environmental constraints.
- Ignores transmission network constraints.
- Assumes marginal costs remain constant over the planning horizon.
- Assumes constant startup costs. (Startup costs are often modeled by the approximationo of an exponential function of time)
- Assumes zero cool-down costs
- Assumes zero No-load costs

In a real-life UCP some of the above features can cause non-trivial complications. However, our simplified example will serve us well for illustrative purposes.

# A simple economic dispatch model with LP

Let us start by making some relaxations to our problem (in subsequent sections, we will strike out these relaxations one by one).
1. All units are running and available at all times (i.e. we don't have the choice to turn a unit ON/OFF.)
2. Startup costs are ignored.
3. Minimum up- and down-time constraints do not apply.

 Now, we are interested in finding the optimal power output of each unit in each time period that minimizes the total marginal cost.

Parameters:

$c_j$ := Marginal cost of running unit $j$ (\$/MWh)

$D_t$ := Total demand in time slot $t$ (MW)

$P_{j, min}$ := Minimum recommended power output for unit $j$ (MW)

$P_{j, max}$ := Maximum recommended power output for unit $j$ (MW)

$J$ := Indexed set of all generators (a.k.a. units)

$T$ := Indexed set of all time slots/periods (a.k.a. planning horizon)

Variables:

$p_{j,t}$ := Power output of unit $j$ in time slot $t$ (MW)

Since we are only considering the marginal costs of runnint the units, the objective is pretty straightforward

\begin{equation*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}
\end{equation*}

There are two sets of constraints we need to consider. The power output of each generating unit must be within the recommended output range. And the total power output of all generating units must satisfy the demand in each time period. We can write these constraints as follows:

\begin{gather*}
P_{j, min} \le p_{j, t} \le P_{j, max}\ \text{(Output Range)}\\
\sum_{j \in J} p_{j, t} \ge D_t\ \text{(Demand)}\\
\end{gather*}

We get the following model

\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}\\\tag{LP}
     \text{s.t.}\quad P_{j, min} \le p_{j, t} \le P_{j, max}\\
     \sum_{j \in J} p_{j, t} \ge D_t
\end{gather*}

Let us solve (LP) using `cvxpy`.


```python
import cvxpy as cp
import cvxopt as cv
import numpy as np
import pandas as pd

from solver.utils import get_result_summary, prettify

# parameters
cA = 10; cB = 12; cC = 20;

p_min_A = 150
p_min_B = 50
p_min_C = 10

p_max_A = 250
p_max_B = 100
p_max_C = 50

D1 = 150; D2 = 300; D3 = 200

T = 3
```


```python
#variables
pA1 = cp.Variable(1, nonneg=True, name='pA1')
pA2 = cp.Variable(1, nonneg=True, name='pA2')
pA3 = cp.Variable(1, nonneg=True, name='pA3')

pB1 = cp.Variable(1, nonneg=True, name='pB1')
pB2 = cp.Variable(1, nonneg=True, name='pB2')
pB3 = cp.Variable(1, nonneg=True, name='pB3')

pC1 = cp.Variable(1, nonneg=True, name='pC1')
pC2 = cp.Variable(1, nonneg=True, name='pC2')
pC3 = cp.Variable(1, nonneg=True, name='pC3')
```


```python
obj_LP = cp.Minimize(
    cA * (pA1 + pA2 + pA3) +
    cB * (pB1 + pB2 + pB3) +
    cC * (pC1 + pC2 + pC3)
)
```


```python
LP_output_range_cons = [
    pA1 >= p_min_A,
    pA1 <= p_max_A,
    pA2 >= p_min_A,
    pA2 <= p_max_A,
    pA3 >= p_min_A,
    pA3 <= p_max_A,
    
    pB1 >= p_min_B,
    pB1 <= p_max_B,
    pB2 >= p_min_B,
    pB2 <= p_max_B,
    pB3 >= p_min_B,
    pB3 <= p_max_B,
    
    pC1 >= p_min_C,
    pC1 <= p_max_C,
    pC2 >= p_min_C,
    pC2 <= p_max_C,
    pC3 >= p_min_C,
    pC3 <= p_max_C,
]

LP_demand_cons = [
    pA1 + pB1 + pC1 >= D1,
    pA2 + pB2 + pC2 >= D2,
    pA3 + pB3 + pC3 >= D3,
]
 
cons_LP = LP_output_range_cons + LP_demand_cons
```


```python
LP = cp.Problem(obj_LP, cons_LP)
```


```python
LP.solve();
```


```python
summary_LP = get_result_summary(LP)
```


```python
summary_LP['status']
```




    'optimal'




```python
summary_LP['optimal_value']
```




    7799.999998684703




```python
prettify(summary_LP['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pA1</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pA2</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pA3</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pB1</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pB2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pB3</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pC1</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pC2</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pC3</td>
      <td>10.0</td>
    </tr>
  </tbody>
</table>
</div>



The minimum marginal cost of running the three units over the horizon is <mark> $7799.99 </mark>. Also note that unit A runs at minimum capacity during periods 1 and 3. During the peak period 2, the output of unit A is increased to meet the increased demand while units B and C continue to run at minimum capacity throughout the horizon. This makes sense because units B and C have a higher marginal cost than unit A. Next, we consider a model which allows us to choose which units are ON during each slot.

# A first MIP formulation

Here, we get rid of the first relaxation:
1. ~All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.~
2. Startup costs are ignored.
3. Minimum up- and down-time constraints do not apply.

We now have the choice to turn one or more units ON/OFF during any of the time slots. We will use a set of binary variables to model this choice.

\begin{equation*}
u_{j, t} :=
\begin{cases}
   1 &\text{if unit $j$ is ON in slot $t$}\\
   0 &\text{otherwise }
\end{cases}
\end{equation*}

Our objective as well as the demand constraint remain unchanged. However, the output range constraint needs to be modified in order to incorporate the new $u_{j, t}$ variables.

\begin{align*}
P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\ \text{(Output Range)}\\
\end{align*}

Note that in the above formulation, if the unit $j$ is OFF in period $t$, the power output $p_{j, t}$ is forced to be zero, so that we don't have any power output contribution from an OFF unit. Below is the new MIP model



\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t}\\ \tag{MIP-1}
     \text{s.t.}\quad P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\\ 
     \sum_{j \in J} p_{j, t} \ge D_t\\
     u_{j, t} \in \{0, 1\} \ \forall\ j, t
\end{gather*}

Let's solve (MIP-1) with `cxvpy`


```python
# ON/OFF variables
uA1 = cp.Variable(1, boolean=True, name='uA1')
uA2 = cp.Variable(1, boolean=True, name='uA2')
uA3 = cp.Variable(1, boolean=True, name='uA3')

uB1 = cp.Variable(1, boolean=True, name='uB1')
uB2 = cp.Variable(1, boolean=True, name='uB2')
uB3 = cp.Variable(1, boolean=True, name='uB3')

uC1 = cp.Variable(1, boolean=True, name='uC1')
uC2 = cp.Variable(1, boolean=True, name='uC2')
uC3 = cp.Variable(1, boolean=True, name='uC3')
```


```python
# Objective remains unchanged
obj_MIP1 = LP.objective
```


```python
# New constraints
MIP1_output_range_cons = [
    # output range
    pA1 >= p_min_A * uA1,
    pA1 <= p_max_A * uA1,
    pA2 >= p_min_A * uA2,
    pA2 <= p_max_A * uA2,
    pA3 >= p_min_A * uA3,
    pA3 <= p_max_A * uA3,
    
    pB1 >= p_min_B * uB1,
    pB1 <= p_max_B * uB1,
    pB2 >= p_min_B * uB2,
    pB2 <= p_max_B * uB2,
    pB3 >= p_min_B * uB3,
    pB3 <= p_max_B * uB3,
    
    pC1 >= p_min_C * uC1,
    pC1 <= p_max_C * uC1,
    pC2 >= p_min_C * uC2,
    pC2 <= p_max_C * uC2,
    pC3 >= p_min_C * uC3,
    pC3 <= p_max_C * uC3,
]

cons_MIP1 = MIP1_output_range_cons + LP_demand_cons # demand constraints don't change
```


```python
MIP1 = cp.Problem(obj_MIP1, cons_MIP1)
```


```python
MIP1.solve();
```


```python
summary_MIP1 = get_result_summary(MIP1)
```


```python
summary_MIP1['status']
```




    'optimal'




```python
summary_MIP1['optimal_value']
```




    6600.0




```python
prettify(summary_MIP1['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pA1</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pA2</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pA3</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pB2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pB3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pC1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pC2</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pC3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>uA1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>uA2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>uA3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>uB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>uB2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>uB3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>uC1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>uC2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>uC3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The optimal cost with the basic MIP model turns out to be <mark>$6600</mark>. Note that this is <mark>\$1200 cheaper</mark> than the result of the (LP) model. Indeed, being able to choose which units to commit during each time slot has saved us some money!

Note that the model chose to fulfill all of the demand in period 1 with unit A, which is the cheapest to run. Both unit B and unit C are more expensive to run and were kept OFF during this period. For the peak demand period, unit B was turned ON to meet the additional demand and was turned back off at the end of the peak period. Unit C, which is the most expensive to run, was never turned ON.

# An MIP model with startup costs

Here, we get rid of the second relaxation:
1. ~All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.~
2. ~Startup costs are ignored.~
3. Minimum up- and down-time constraints do not apply.

We now consider the startup costs of the units. A unit incurs a startup cost in a time period only if it was started up in that period. We need a binary variable to indicate if a unit was turnd ON in a given period. If so, the unit will incur the startup cost in that period in addition to the marginal cost. (For simplicity, we have assumed the cool-down costs to be $0$.)

We introduce a new parameter to denote the startup costs of the units

\begin{align*}
    c_j^u := \text{Startup cost of unit } j\ (\$)
\end{align*}

and a new binary variable

\begin{equation*}
\alpha_{j, t} :=
\begin{cases}
   1 &\text{if unit $j$ was started in period $t$}\\
   0 &\text{otherwise }
\end{cases}
\end{equation*}

The new objective is given by:

\begin{align*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t} + \alpha_{j, t} c_j^u\\
\end{align*}

The demand constraint as well as the output range constraints remain the same as for (MIP-1). However, we need a new constraint in order to ensure that $\alpha_{j, t} = 1$ if and only if unit $j$ was started up in period $t$. This constraint can be modelled by the below function

\begin{align*}
    \alpha_{j, t} = \lfloor \frac{u_{j, t} - u_{j, t-1} + 1}{2} \rfloor
\end{align*}

The above non-linear function can be expressed in terms of linear constraints as follows:

\begin{align*}
    \alpha_{j, t} &\le \frac{u_{j, t} - u_{j, t-1} + 1}{2},\ \ \ \  \alpha_{j, t} + 1 \ge \frac{u_{j, t} - u_{j, t-1} + 1}{2} + .25\qquad \text{(Startup)}
\end{align*}

Our new model can be written as:

\begin{gather*}
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t} + \alpha_{j, t} c_j^u\\
     \text{s.t.}\quad P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\\ 
     \sum_{j \in J} p_{j, t} \ge D_t\\ \tag{MIP-2}
     \alpha_{j, t} \le \frac{u_{j, t} - u_{j, t-1} + 1}{2}\\
     \alpha_{j, t} + 1 \ge \frac{u_{j, t} - u_{j, t-1} + 1}{2} + .25\\
     u_{j, t} \in \{0, 1\} \ \forall\ j, t\\
     \alpha_{j, t} \in \{0, 1\} \ \forall\ j, t
\end{gather*}

Lets solve (MIP-2) with `cvxpy`


```python
# parameters
## startup costs
cA_up = 1000
cB_up = 600
cC_up = 100

## initial states
uA0 = 1 # ON
uB0 = 0 # OFF
uC0 = 0 # OFF
```


```python
# startup variables
alpha_A1 = cp.Variable(1, boolean=True, name='alpha_A1')
alpha_A2 = cp.Variable(1, boolean=True, name='alpha_A2')
alpha_A3 = cp.Variable(1, boolean=True, name='alpha_A3')


alpha_B1 = cp.Variable(1, boolean=True, name='alpha_B1')
alpha_B2 = cp.Variable(1, boolean=True, name='alpha_B2')
alpha_B3 = cp.Variable(1, boolean=True, name='alpha_B3')


alpha_C1 = cp.Variable(1, boolean=True, name='alpha_C1')
alpha_C2 = cp.Variable(1, boolean=True, name='alpha_C2')
alpha_C3 = cp.Variable(1, boolean=True, name='alpha_C3')
```


```python
# objective
obj_MIP2 = cp.Minimize(
    cA * (pA1 + pA2 + pA3) + cA_up * (alpha_A1 + alpha_A2 + alpha_A3) +
    cB * (pB1 + pB2 + pB3) + cB_up * (alpha_B1 + alpha_B2 + alpha_B3) +
    cC * (pC1 + pC2 + pC3) + cC_up * (alpha_C1 + alpha_C2 + alpha_C3)
)
```


```python
# constraints
MIP2_startup_cons =  [
    alpha_A1 <= (uA1 - uA0 + 1)/2,
    alpha_A1 >= (uA1 - uA0 + 1)/2 - .75,
    alpha_A2 <= (uA2 - uA1 + 1)/2,
    alpha_A2 >= (uA2 - uA1 + 1)/2 - .75,
    alpha_A3 <= (uA3 - uA2 + 1)/2,
    alpha_A3 >= (uA3 - uA2 + 1)/2 - .75,
    
    alpha_B1 <= (uB1 - uB0 + 1)/2,
    alpha_B1 >= (uB1 - uB0 + 1)/2 - .75,
    alpha_B2 <= (uB2 - uB1 + 1)/2,
    alpha_B2 >= (uB2 - uB1 + 1)/2 - .75,
    alpha_B3 <= (uB3 - uB2 + 1)/2,
    alpha_B3 >= (uB3 - uB2 + 1)/2 - .75,
    
    alpha_C1 <= (uC1 - uC0 + 1)/2,
    alpha_C1 >= (uC1 - uC0 + 1)/2 - .75,
    alpha_C2 <= (uC2 - uC1 + 1)/2,
    alpha_C2 >= (uC2 - uC1 + 1)/2 - .75,
    alpha_C3 <= (uC3 - uC2 + 1)/2,
    alpha_C3 >= (uC3 - uC2 + 1)/2 - .75,
]
    
cons_MIP2 = MIP1.constraints + MIP2_startup_cons
```


```python
MIP2 = cp.Problem(obj_MIP2, cons_MIP2)
MIP2.solve();
summary_MIP2 = get_result_summary(MIP2)
```


```python
summary_MIP2['status']
```




    'optimal'




```python
summary_MIP2['optimal_value']
```




    7100.0




```python
prettify(summary_MIP2['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alpha_A1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alpha_A2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alpha_A3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alpha_B1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alpha_B2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alpha_B3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>alpha_C1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alpha_C2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alpha_C3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pA1</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pA2</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pA3</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>pB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pB2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pB3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pC1</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>pC2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>pC3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>uA1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>uA2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>uA3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>uB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>uB2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>uB3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>uC1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>uC2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>uC3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The optimal value of the total cost (setup + marginal) is <mark> $7100</mark>. This optimal value is greater than the optimal value of MIP1 (\$6600) because it includes the setup cost which we ignored in MIP1.
    
Also note that, once again, the model chose to meet all of the demand in period 1 with unit A. The reason for this is two-fold. First, unit A is the cheapest to run and incurrs the smallest marginal cost. So, it makes sense to preferentially run unit A whenever possible. Second, unit A was already in ON state at the beginning of the planning horizon and did not need to be turned ON. As a result, by keeping unit A running, we avoided the setup cost for unit A.

We should also note that during the peak demand period, the model chose to turn on unit C instead of unit B as it had done in MIP1. This is because while unit B is cheaper to run than unit C, it is more expensive to start up than unit C. Since the setup cost dominates the marginal cost, the model preferred to start up unit C.

## An MIP model with minimum up- and down-time constraints

Here, we get rid of the third and final relaxation:
1. ~~All units are running and available at all times. I.e. we don't have the choice to turn a unit ON/OFF.~~
2. ~~Startup costs are ignored.~~
3. ~~Minimum up- and down-time constraints do not apply.~~

We now consider the up- and down-time constraints that apply to each unit. In order to model these constraints we need need to define two new parameters:

$$
UT_j := \text{Minimum length of time (in periods/slots) that unit j must remain ON before it can be turned OFF.}\\
DT_j := \text{Minimum length of time (in periods/slots) that unit j must remain OFF before it can be turned ON.}
$$


```python
# parameters
UT_A = 3; UT_B = 2; UT_C = 1;
DT_A = 3; DT_B = 1; DT_C = 1;
```

We also need a new variable to indicate if a given unit was stopped in a given period.

$$
\beta_{j, t} :=
\begin{cases}
   1 &\text{   if unit j was stopped in period t}\\
   0 &\text{   otherwise }
\end{cases}
$$


```python
# variables
beta_A1 = cp.Variable(1, boolean=True, name="beta_A1")
beta_A2 = cp.Variable(1, boolean=True, name="beta_A2")
beta_A3 = cp.Variable(1, boolean=True, name="beta_A3")

beta_B1 = cp.Variable(1, boolean=True, name="beta_B1")
beta_B2 = cp.Variable(1, boolean=True, name="beta_B2")
beta_B3 = cp.Variable(1, boolean=True, name="beta_B3")

beta_C1 = cp.Variable(1, boolean=True, name="beta_C1")
beta_C2 = cp.Variable(1, boolean=True, name="beta_C2")
beta_C3 = cp.Variable(1, boolean=True, name="beta_C3")
```

We need to specify the shutdown constraint

$$
\beta_{j, t} = \lfloor \frac{(-u_{j, t} + u_{j, t-1} + 1)}{2} \rfloor
$$

which can be expressed in the form of linear inequalities as follows:

$$
\beta_{j, t} \le \frac{-u_{j, t} + u_{j, t-1} + 1}{2},\ \ \ \  \beta_{j, t} + 1 \ge \frac{-u_{j, t} + u_{j, t-1} + 1}{2} + .25\qquad \text{(Shut-down)}
$$


```python
MIP3_shutdown_cons = [
    beta_A1 <= (-uA1 + uA0 + 1)/2,
    beta_A1 >= (-uA1 + uA0 + 1)/2 + .25,
    beta_A2 <= (-uA2 + uA1 + 1)/2,
    beta_A2 >= (-uA2 + uA1 + 1)/2 + .25,
    beta_A3 <= (-uA3 + uA2 + 1)/2,
    beta_A3 >= (-uA3 + uA2 + 1)/2 + .25,

    beta_B1 <= (-uB1 + uB0 + 1)/2,
    beta_B1 >= (-uB1 + uB0 + 1)/2 + .25,
    beta_B2 <= (-uB2 + uB1 + 1)/2,
    beta_B2 >= (-uB2 + uB1 + 1)/2 + .25,
    beta_B3 <= (-uB3 + uB2 + 1)/2,
    beta_B3 >= (-uB3 + uB2 + 1)/2 + .25,
    
    beta_C1 <= (-uC1 + uC0 + 1)/2,
    beta_C1 >= (-uC1 + uC0 + 1)/2 + .25,
    beta_C2 <= (-uC2 + uC1 + 1)/2,
    beta_C2 >= (-uC2 + uC1 + 1)/2 + .25,
    beta_C3 <= (-uC3 + uC2 + 1)/2,
    beta_C3 >= (-uC3 + uC2 + 1)/2 + .25,
]
```

We also need the minimum uptime constraints:

$$
\tag{Up-time}
\sum_{i=t}^{t + UT_j - 1} u_{j, i} \ge \alpha_{j,t} UT_j \ ,\qquad \forall\ t \in \{1,\ T - UT_j + 1\}\\
\sum_{i=t}^T u_{j, i} \ge \alpha_{j, t} (T - t + 1) \ ,\qquad \forall\ t \in \{T - UT_j + 2,\ T\}
$$




```python
def generate_uptime_cons(j, t):
    sum = ''
    UT_j = eval('UT_{j}'.format(j=j))
    if t <= T - UT_j + 1:
        for i in range(t, t + UT_j):
            sum += 'u{j}{i} +'.format(j=j, i=i)
        sum = sum[:-1]
        sum += ' >= alpha_{j}{t} * UT_{j}'.format(j=j, t=t)
    else:
        for k in range(t, T + 1):
            sum += 'u{j}{k} +'.format(j=j, k=k)
        sum = sum[:-1]
        sum += ' >= alpha_{j}{t} * (T - {t} + 1)'.format(j=j, t=t)
    return eval(sum)
```


```python
MIP3_uptime_cons = []
for j in ['A', 'B', 'C']:
    for t in range(1, T + 1):
        cons = generate_uptime_cons(j, t)
        print(cons)
        MIP3_uptime_cons.append(cons)
```

    alpha_A1 @ 3.0 <= uA1 + uA2 + uA3
    alpha_A2 @ 2.0 <= uA2 + uA3
    alpha_A3 @ 1.0 <= uA3
    alpha_B1 @ 2.0 <= uB1 + uB2
    alpha_B2 @ 2.0 <= uB2 + uB3
    alpha_B3 @ 1.0 <= uB3
    alpha_C1 @ 1.0 <= uC1
    alpha_C2 @ 1.0 <= uC2
    alpha_C3 @ 1.0 <= uC3


And the minimum downtime constraints:

$$
\tag{Down-time}
\sum_{i=t}^{t + DT_j - 1} (1 - u_{j, i}) \ge \beta_{j,t} DT_j \ ,\qquad \forall\ t \in \{1,\ T - DT_j + 1\}\\
\sum_{i=t}^T (1 - u_{j, i}) \ge \beta_{j, t} (T - t + 1) \ ,\qquad \forall\ t \in \{T - DT_j + 2,\ T\}
$$


```python
def generate_downtime_cons(j, t):
    sum = ''
    DT_j = eval('DT_{j}'.format(j=j))
    if t <= T - DT_j + 1:
        for i in range(t, t + DT_j):
            sum += '(1 - u{j}{i}) +'.format(j=j, i=i)
        sum = sum[:-1]
        sum += ' >= beta_{j}{t} * DT_{j}'.format(j=j, t=t)
    else:
        for k in range(t, T + 1):
            sum += '(1 - u{j}{k}) +'.format(j=j, k=k)
        sum = sum[:-1]
        sum += ' >= beta_{j}{t} * (T - {t} + 1)'.format(j=j, t=t)
    return eval(sum)
```


```python
MIP3_downtime_cons = []
for j in ['A', 'B', 'C']:
    for t in range(1, T + 1):
        cons = generate_downtime_cons(j, t)
        print(cons)
        MIP3_downtime_cons.append(cons)
```

    beta_A1 @ 3.0 <= 1.0 + -uA1 + 1.0 + -uA2 + 1.0 + -uA3
    beta_A2 @ 2.0 <= 1.0 + -uA2 + 1.0 + -uA3
    beta_A3 @ 1.0 <= 1.0 + -uA3
    beta_B1 @ 1.0 <= 1.0 + -uB1
    beta_B2 @ 1.0 <= 1.0 + -uB2
    beta_B3 @ 1.0 <= 1.0 + -uB3
    beta_C1 @ 1.0 <= 1.0 + -uC1
    beta_C2 @ 1.0 <= 1.0 + -uC2
    beta_C3 @ 1.0 <= 1.0 + -uC3


Finally, we need the logical constraints [1] to ensure that $\alpha_{j,t} = 1$ only when the unit is scheduled to be switched on in slot $t$ (i.e., $u_{j, t-1} = 0$ and $u_{j, t} = 1$), and $\beta_{j, t} = 1$ only when the unit is scheduled to be switched off in slot $t$ (i.e., $u_{j, t-1} = 1$ and $u_{j, t} = 0$).

$$
u_{j, t-1} - u_{j, t} + \alpha_{j, t} - \beta_{j, t} = 0\ ,\qquad \forall\ t\in T,\ j\in J \tag{Logical}
$$


```python
MIP3_logical_cons = []
for j in ['A', 'B', 'C']:
    for t in range(1, T + 1):
        cons = eval('u{j}{t_1} - u{j}{t} + alpha_{j}{t} - beta_{j}{t} == 0'.format(j=j, t_1=t-1, t=t))
        print(cons)
        MIP3_logical_cons.append(cons)
```

    1.0 + -uA1 + alpha_A1 + -beta_A1 == 0.0
    uA1 + -uA2 + alpha_A2 + -beta_A2 == 0.0
    uA2 + -uA3 + alpha_A3 + -beta_A3 == 0.0
    0.0 + -uB1 + alpha_B1 + -beta_B1 == 0.0
    uB1 + -uB2 + alpha_B2 + -beta_B2 == 0.0
    uB2 + -uB3 + alpha_B3 + -beta_B3 == 0.0
    0.0 + -uC1 + alpha_C1 + -beta_C1 == 0.0
    uC1 + -uC2 + alpha_C2 + -beta_C2 == 0.0
    uC2 + -uC3 + alpha_C3 + -beta_C3 == 0.0


Since we are ignoring the shutdown costs of the units, our objective remains the same as that in (MIP-2). Below is our final model:

$$
    \text{minimize} \sum_{t\in T} \sum_{j \in J} c_j p_{j, t} + \alpha_{j, t} c_j^u
$$

$$
     \text{s.t.}\quad P_{j, min} u_{j, t} \le p_{j, t} \le P_{j, max} u_{j, t}\quad \text{(Output Range)}
$$

$$
     \sum_{j \in J} p_{j, t} \ge D_t\quad \text{(Demand)} 
$$

$$
     \alpha_{j, t} \le \frac{u_{j, t} - u_{j, t-1} + 1}{2},\ \alpha_{j, t} + 1 \ge \frac{u_{j, t} - u_{j, t-1} + 1}{2} + .25 \quad \text{(Startup)}
$$

$$
\sum_{i=t}^{t + UT_j - 1} u_{j, i} \ge \alpha_{j,t} UT_j \ ,\ \forall\ t \in \{1,\ T - UT_j + 1\} \quad \text{(Uptime)}
$$

$$
\sum_{i=t}^T u_{j, i} \ge \alpha_{j, t} (T - t + 1) \ ,\ \ \forall\ t \in \{T - UT_j + 2,\ T\} \quad \text{(Uptime)}
$$

$$
\sum_{i=t}^{t + DT_j - 1} (1 - u_{j, i}) \ge \beta_{j,t} DT_j \ ,\ \forall\ t \in \{1,\ T - DT_j + 1\} \quad \text{(Downtime)}
$$

$$
\sum_{i=t}^T (1 - u_{j, i}) \ge \beta_{j, t} (T - t + 1) \ ,\ \forall\ t \in \{T - DT_j + 2,\ T\}\quad \text{(Downtime)}
$$
$$
u_{j, t-1} - u_{j, t} + \alpha_{j, t} - \beta_{j, t} = 0\ ,\qquad \forall\ t\in T,\ j\in J \quad \text{(Logical)}
$$

$$
     u_{j, t} \in \{0, 1\} \ \forall\ j, t\\
     \alpha_{j, t} \in \{0, 1\} \ \forall\ j, t\\
     \beta_{j, t} \in \{0, 1\} \ \forall\ j, t \tag{MIP-3}
$$

We now solve (MIP-3) with `cvxpy`


```python
MIP3_obj = MIP2.objective
```


```python
MIP3_cons = MIP2.constraints + MIP3_uptime_cons + MIP3_downtime_cons + MIP3_logical_cons
# + MIP3_shutdown_cons
```


```python
MIP3 = cp.Problem(MIP3_obj, MIP3_cons)
MIP3.solve();
summary_MIP3 = get_result_summary(MIP3);
```


```python
summary_MIP3['status']
```




    'optimal'




```python
summary_MIP3['optimal_value']
```




    7100.0




```python
prettify(summary_MIP3['optimal_solution'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alpha_A1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>alpha_A2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>alpha_A3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>alpha_B1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>alpha_B2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>alpha_B3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>alpha_C1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>alpha_C2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alpha_C3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>beta_A1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>beta_A2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>beta_A3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>beta_B1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>beta_B2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>beta_B3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>beta_C1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>beta_C2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>beta_C3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>pA1</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>pA2</td>
      <td>250.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pA3</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>pB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>pB2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>pB3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>pC1</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>pC2</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>pC3</td>
      <td>-0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>uA1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>uA2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>uA3</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>uB1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>uB2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>uB3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>uC1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>uC2</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>uC3</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



As it turns out, the optimal cost of scheduling the three units over the given planning horizon remains unchanged at <mark>$7100</mark>. What is interesting is that we are able to satisfy multiple additional constraints at the same total cost as MIP2! This fact is due to the (simplified) structure of our specific problem. In a real-life scenario, adding constraints can lead to increased total cost. Nevertheless, discussing the above porblem gives us insight into how MIP can be used to model and solve similar, larger problems at scale.

# References

[1] https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/298917/UnitCommitment.pdf?sequence=3

[2] https://www.youtube.com/watch?v=jS15dU_422Q
