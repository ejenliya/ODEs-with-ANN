#%%
import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from time import process_time

import plotly.graph_objects as go
#%%
def sigmoid(z): 
    return 1 / (1 + np.exp(-z))

def neural_network(parameters , x_in):
    n_hidden = np.size(np.array(parameters, dtype='object')) - 1
    num_values = np.size(x_in)
    x_in = x_in.reshape(-1, num_values)
    x_input = x_in
    x_prev = x_input

    for i in range(n_hidden):
        w_hidden = parameters[i]
        x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)
        z_hidden = np.matmul(w_hidden , x_prev)
        x_hidden = sigmoid(z_hidden)
        x_prev = x_hidden

    w_output = parameters[-1]
    x_prev = np.concatenate((np.ones((1, num_values)), x_prev), axis=0)
    z_output = np.matmul(w_output , x_prev)
    x_output = z_output
    return x_output

# hardcode
def g_trial_solution(x, parameters):
    # return 1 + (x ** 2) * neural_network(parameters , x) # example 3
    # return 1 + x * neural_network(parameters , x) # du/dt = 1/u
    # return 2 + x * neural_network(parameters , x) #du/dt = u
    return 1 + x + (x ** 2) * neural_network(parameters , x) # example 4

def cost_function(parameters , x_in):
    # hardcoded right side of ode
    def f(x, trial): 
        # return 1 / trial #du/dt = 1/u
        # return trial # du/dt = u
        # return -trial # example 3
        return np.tan(x) - trial #example 4

    g_t = g_trial_solution(x_in, parameters)
    #d_g_t = elementwise_grad(g_trial_solution , 0)(x_in, parameters)
    d2_g_t = elementwise_grad(elementwise_grad(g_trial_solution , 0))(x_in, parameters) # second derivative
    right_side = f(x_in, g_t)

    err_sqr = (d2_g_t - right_side) ** 2
    cost_sum = np.sum(err_sqr)
    return cost_sum / np.size(err_sqr)  # np.square(np.subtract(d2_g_t , right_side)).mean()

losses = []
def neural_network_solve_ode(x_in, num_neurons , num_iterations , learning_rate):
    n_hidden = np.size(num_neurons)
    parameters = [None] * (n_hidden + 1)
    parameters[0] = npr.randn(num_neurons[0], 2)

    for j in range(1, n_hidden):
        parameters[j] = npr.randn(num_neurons[j], num_neurons[j - 1] + 1)

    parameters[-1] = npr.randn(1, num_neurons[-1] + 1)  
    print('Initial cost: %g' % cost_function(parameters , x_in))    
    cost_function_grad = grad(cost_function , 0)

    for _ in range(num_iterations):
        cost_grad = cost_function_grad(parameters , x_in)
        losses.append(cost_function(parameters , x_in))
        for j in range(n_hidden + 1):
            parameters[j] = parameters[j] - learning_rate * cost_grad[j]
    print('Final cost: %g' % cost_function(parameters , x_in))
    return parameters

# hardcode only for plotting
def g_exact(x): 
    return np.cos(x) + 2 * np.sin(x) - np.cos(x) * np.log((1 / np.cos(x)) + np.tan(x)) # example 4
    # return np.cos(x) # example 3
    # return (2 * x + 1) ** (1 / 2)    #du/dt = 1/u
    # return 2 * np.exp(x)   #du/dt = u

# =========== euler part ===============
def implicit_euler_residual(yp, ode, to, yo, tp):
    return  yp - yo - (tp - to) * ode(tp, yp)

def implicit_euler(ode, tspan=np.array([0.0, 1.0]), y0=[1, 1], num_steps=10):
    if np.ndim(y0) == 0:
        m = 1
    else:
        m = len(y0)

    t = np.zeros(num_steps + 1)
    y = np.zeros([num_steps + 1, m])
    dt = (tspan[1] - tspan[0]) / float(num_steps)
    t[0] = tspan[0]
    y[0, :] = y0
    for i in range(0, num_steps):
        to = t[i]
        yo = y[i, :]
        tp = t[i] + dt
        yp = yo + dt * ode(to, yo)

        yp = fsolve(implicit_euler_residual , yp, args=(ode, to, yo, tp))
        t[i + 1] = tp
        y[i + 1, :] = yp[:]

    return y, t

#%% main
npr.seed(4155)
nx = 10
x = np.linspace(0, 1, nx)
num_hidden_neurons = [100, 50]
num_iter = 1000
lmb = 1e-3

t1_start = process_time()
P = neural_network_solve_ode(x, num_hidden_neurons , num_iter , lmb)
t1_stop = process_time()

g_dnn_ag = g_trial_solution(x, P)
g_analytical = g_exact(x)

#%%
tnum_start = process_time()
# du/dt = 1/u 
#s, t = implicit_euler(lambda t, s: 1 / s, tspan=np.array([0.0, 1.0]), y0=1, num_steps=nx)
# du/dt = u 
# s, t = implicit_euler(lambda t, s: s, tspan=np.array([0.0, 1.0]), y0=2, num_steps=nx)
tnum_stop = process_time()

#%% euler fixed 
import numpy as np
def system_deriv(t, rf):
    r = rf[0]
    f = rf[1]
    drdt = f
    dfdt = np.tan(t) - r
    drfdt = np.array([drdt, dfdt])
    return drfdt

def implicit_euler_step(rf, t, dt):
    def func(new_rf):
        return new_rf - rf - dt * system_deriv(t + dt, new_rf)
    new_rf = fsolve(func, rf)
    return new_rf

def solve_system(t0, t1, rf0, dt):
    t_values = np.arange(t0, t1 + dt, dt)
    rf_values = np.zeros((len(t_values), 2))
    rf_values[0] = rf0
    for i in range(1, len(t_values)):
        rf_values[i] = implicit_euler_step(rf_values[i - 1], t_values[i - 1], dt)
    return t_values, rf_values

t0 = 0.0
t1 = 1.0
rf0 = np.array([1.0, 1.0])
dt = 0.1

tnum_start = process_time()
t_values, rf_values = solve_system(t0, t1, rf0, dt)
tnum_stop = process_time()


#%%
max_diff = np.max(np.abs(g_dnn_ag - g_analytical))
print("MAD NN: %g" %max_diff)
print("Time NN: ", t1_stop - t1_start)
max_diff_num = np.max(np.abs(rf_values[1:, 0] - g_analytical))
mse = np.square(rf_values[1:, 0] - g_analytical).mean()
print("MSE Euler: ", mse)
print("MAD Euler: %g" % max_diff_num)
print("Time Euler: ", tnum_stop - tnum_start)


#%%
trace_exact = go.Scatter(x=x, y=g_analytical, mode='markers+lines', name='Exact u')
trace_dnn = go.Scatter(x=x, y=g_dnn_ag[0], mode='lines', name='ANN u')
trace_euler = go.Scatter(x=t_values, y=rf_values[:, 0], mode='lines', name='Euler u')
#trace_euler = go.Scatter(x=t, y=s.flatten(), mode='lines', name='Euler u')

layout = go.Layout(
    xaxis=dict(title='t'),
    yaxis=dict(title='u(t)'),
    legend=dict(x=0, y=1)
)
fig = go.Figure(data=[trace_exact, trace_dnn, trace_euler], layout=layout)
fig.show()
# %%