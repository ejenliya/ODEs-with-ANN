#%%
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, SinActv

from scipy.integrate import odeint
import autograd.numpy as np
import torch
from time import process_time
import plotly.graph_objects as go
#%%
def get_system1(ts1):
    alpha, beta, delta, gamma = 1.5, 1, 1, 1
    def ode_system1(u, v, t): 
        return [diff(u, t) - (alpha*u  - beta*u*v),
                diff(v, t) - (delta*u*v - gamma*v),]

    conditions1 = [IVP(t_0=0.0, u_0=1.5), IVP(t_0=0.0, u_0=1.0)]
    nets = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([200, 100]), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([200, 100]), actv=SinActv)
    ]

    solver1 = Solver1D(ode_system1, conditions1, t_min=0.1, t_max=12.0, nets=nets)
    solver1.fit(max_epochs = 2000)
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

ts1 = np.linspace(0, 12, 100)
prey_net, pred_net, prey_num, pred_num = get_system1(ts1)
#%%
system1 = go.Figure()
system1.add_trace(go.Scatter(x=ts1, y=prey_net, mode='lines', name='ANN u'))
system1.add_trace(go.Scatter(x=ts1, y=pred_net, mode='lines', name = 'ANN v'))
system1.add_trace(go.Scatter(x=ts1, y=prey_num, mode='markers', name = 'Exact u'))
system1.add_trace(go.Scatter(x=ts1, y=pred_num, mode='markers', name = 'Exact v'))
system1.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=1))
#%%
print('MSE of model 1: ', np.mean((prey_net - prey_num)**2 + (pred_net - pred_num)**2))
print('MAD of model 1: ', np.max([np.max(np.abs(prey_net - prey_num)), np.max(np.abs(pred_net - pred_num))]))






# =============================== model 2 ========================

#%%
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
    solver2.fit(max_epochs=2000)
    solution2 = solver2.get_solution()

    net_sol1, net_sol2 = solution2(ts2, to_numpy=True)
    return net_sol1, net_sol2

ts2 = np.linspace(0, 3, 50)
net_sol1, net_sol2 = get_system2(ts2)
#%%
system2 = go.Figure()
system2.add_trace(go.Scatter(x=ts2, y=net_sol1, mode='lines', name='ANN u'))
system2.add_trace(go.Scatter(x=ts2, y=net_sol2, mode='lines', name = 'ANN v'))
system2.add_trace(go.Scatter(x=ts2, y=np.sin(ts2), mode='markers', name = 'Exact u'))
system2.add_trace(go.Scatter(x=ts2, y=1 + ts2**2, mode='markers', name = 'Exact v'))
system2.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=1))
#%%
num_sol1 = np.sin(ts2)
num_sol2 = 1 + ts2**2

print('MSE of model 2: ', np.mean((net_sol1 - num_sol1)**2 + (net_sol2 - num_sol2)**2))
print('MAD of model 2: ', np.max([np.max(np.abs(net_sol1 - num_sol1)), np.max(np.abs(net_sol2 - num_sol2))]))
# %%


# ======================= model 3========================
def get_system3(ts3):

    def ode_system3(u, v, t): 
        return [diff(u, t) - (-2*u + 4*v),
                diff(v, t) - (-u + 3*v)]

    conditions3 = [IVP(t_0=0.0, u_0=3), IVP(t_0=0.0, u_0=0)]
    nets = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([400, 300]), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([400, 300]), actv=SinActv)
    ]

    solver3 = Solver1D(ode_system3, conditions3, t_min=0.1, t_max=1.0, nets=nets)
    solver3.fit(max_epochs = 4000)
    solution3 = solver3.get_solution()

    def dPdt(P, t):
        return [-2*P[0] + 4*P[1], # u = P[0], v = P[1]
                -P[0] + 3*P[1]]
    P0 = [3, 0]
    Ps = odeint(dPdt, P0, ts3)
    prey_num = Ps[:, 0]
    pred_num = Ps[:, 1]

    prey_net, pred_net = solution3(ts3, to_numpy=True)
    return prey_net, pred_net, prey_num, pred_num

ts3 = np.linspace(0, 1, 70)
net_sol1, net_sol2, num_sol1, num_sol2 = get_system3(ts3)
#%%
system3 = go.Figure()
system3.add_trace(go.Scatter(x=ts3, y=net_sol1, mode='lines', name='ANN u'))
system3.add_trace(go.Scatter(x=ts3, y=net_sol2, mode='lines', name = 'ANN v'))
# system3.add_trace(go.Scatter(x=ts3, y=4*np.exp(-ts3) - np.exp(2*ts3), mode='markers', name = 'Exact u'))
# system3.add_trace(go.Scatter(x=ts3, y=np.exp(-ts3) - np.exp(2*ts3), mode='markers', name = 'Exact v'))
system3.add_trace(go.Scatter(x=ts3, y=num_sol1, mode='markers', name = 'Exact u'))
system3.add_trace(go.Scatter(x=ts3, y=num_sol2, mode='markers', name = 'Exact v'))
system3.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=0))
#%%
print('MSE of model 3: ', np.mean((net_sol1 - num_sol1)**2 + (net_sol2 - num_sol2)**2))
print('MAD of model 3: ', np.max([np.max(np.abs(net_sol1 - num_sol1)), np.max(np.abs(net_sol2 - num_sol2))]))

# %%









# ============================== model 4 ===================================
def get_system3(ts3):

    def ode_system3(u, v, t): 
        return [diff(u, t) - (2*u - 5*v + 3),
                diff(v, t) - (5*u - 6*v + 1)]

    conditions3 = [IVP(t_0=0.0, u_0=6), IVP(t_0=0.0, u_0=5)]
    nets = [
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([100, 50]), actv=SinActv),
        FCNN(n_input_units=1, n_output_units=1, hidden_units=([100, 50]), actv=SinActv)
    ]

    solver3 = Solver1D(ode_system3, conditions3, t_min=0.1, t_max=1.5, nets=nets)
    solver3.fit(max_epochs = 2000)
    solution3 = solver3.get_solution()

    def dPdt(P, t):
        return [2*P[0] - 5*P[1] + 3, # u = P[0], v = P[1]
                5*P[0] - 6*P[1] + 1]
    P0 = [6, 5]
    Ps = odeint(dPdt, P0, ts3)
    prey_num = Ps[:, 0]
    pred_num = Ps[:, 1]

    prey_net, pred_net = solution3(ts3, to_numpy=True)
    return prey_net, pred_net, prey_num, pred_num

ts3 = np.linspace(0, 1.5, 70)
net_sol1, net_sol2, num_sol1, num_sol2 = get_system3(ts3)
#%%
system3 = go.Figure()
system3.add_trace(go.Scatter(x=ts3, y=net_sol1, mode='lines', name='ANN u'))
system3.add_trace(go.Scatter(x=ts3, y=net_sol2, mode='lines', name = 'ANN v'))
# system3.add_trace(go.Scatter(x=ts3, y=4*np.exp(-ts3) - np.exp(2*ts3), mode='markers', name = 'Exact u'))
# system3.add_trace(go.Scatter(x=ts3, y=np.exp(-ts3) - np.exp(2*ts3), mode='markers', name = 'Exact v'))
system3.add_trace(go.Scatter(x=ts3, y=num_sol1, mode='markers', name = 'Exact u'))
system3.add_trace(go.Scatter(x=ts3, y=num_sol2, mode='markers', name = 'Exact v'))
system3.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=0))
#%%
print('MSE of model 3: ', np.mean((net_sol1 - num_sol1)**2 + (net_sol2 - num_sol2)**2))
print('MAD of model 3: ', np.max([np.max(np.abs(net_sol1 - num_sol1)), np.max(np.abs(net_sol2 - num_sol2))]))


# %%
