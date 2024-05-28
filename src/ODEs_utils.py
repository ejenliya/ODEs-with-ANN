import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from scipy.optimize import fsolve

EXAMPLE = ''

def set_EXAMPLE(example):
    global EXAMPLE
    EXAMPLE = example

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
    if EXAMPLE == 'first':
        return 2 + x * neural_network(parameters , x)
    elif EXAMPLE == 'second':
        return 1 + x * neural_network(parameters , x)
    else: pass

def cost_function(parameters , x_in):
    # hardcoded right side of ode
    def f(x, trial): 
        if EXAMPLE == 'first':
            return trial
        elif EXAMPLE == 'second':
            return 1 / trial
        else: pass

    g_t = g_trial_solution(x_in, parameters)
    d_g_t = elementwise_grad(g_trial_solution , 0)(x_in, parameters)
    right_side = f(x_in, g_t)

    err_sqr = (d_g_t - right_side) ** 2
    #cost_sum = np.sum(err_sqr)
    return np.square(np.subtract(d_g_t , right_side)).mean()

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
    if EXAMPLE == 'first':
        return 2 * np.exp(x)
    elif EXAMPLE == 'second':
        return (2 * x + 1) ** (1 / 2)
    else: pass


# =========== euler part ===============
def implicit_euler_residual(yp, ode, to, yo, tp):
    return  yp - yo - (tp - to) * ode(tp, yp)

def implicit_euler(ode, tspan=np.array([0.0, 1.0]), y0=2, num_steps=10):
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