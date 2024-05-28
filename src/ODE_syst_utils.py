#%%
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, SinActv

from scipy.integrate import odeint
import autograd.numpy as np
import torch

import plotly.graph_objects as go

def get_system1(ts1):
    alpha, beta, delta, gamma = 1, 1, 1, 1
    def ode_system1(u, v, t): 
        return [diff(u, t) - (alpha*u  - beta*u*v),
                diff(v, t) - (delta*u*v - gamma*v),]

    conditions1 = [IVP(t_0=0.0, u_0=1.5), IVP(t_0=0.0, u_0=1.0)]
    nets = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(200, 100), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=(200, 100), actv=SinActv)
    ]

    solver1 = Solver1D(ode_system1, conditions1, t_min=0.1, t_max=12.0, nets=nets)
    solver1.fit(max_epochs = 1000)
    solution1 = solver1.get_solution()

    def dPdt(P, t):
        return [P[0]*alpha - beta*P[0]*P[1], 
                delta*P[0]*P[1] - gamma*P[1]]
    P0 = [1.5, 1.0]
    Ps = odeint(dPdt, P0, ts1)
    prey_num = Ps[:, 0]
    pred_num = Ps[:, 1]

    prey_net, pred_net = solution1(ts1, to_numpy=True)
    return prey_net, pred_net, prey_num, pred_num

def get_system2(ts2):
    def ode_system2(u, v, t): 
        return [diff(u, t) - torch.cos(t) - u**2 - v + (1 + t**2 + torch.sin(t)),
                diff(v, t) - 2*t + (1 + t**2)*torch.sin(t) - u*v]

    conditions2 = [IVP(t_0=0.0, u_0=0.0), 
                IVP(t_0=0.0, u_0=1.0)]
    nets = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([200, 100]), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([200, 100]), actv=SinActv)
    ]
    solver2 = Solver1D(ode_system2, conditions2, t_min=0.1, t_max=5.0, nets=nets)
    solver2.fit(max_epochs=1000)
    solution2 = solver2.get_solution()

    net_sol1, net_sol2 = solution2(ts2, to_numpy=True)
    return net_sol1, net_sol2
