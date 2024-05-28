import dash
from dash import dcc, html, Input, Output, callback
from flask import Flask
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.figure_factory as ff

import torch
import autograd.numpy as np
import autograd.numpy.random as npr
from time import process_time

#import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint

from ODEs_utils import *
from ODE_syst_utils import *

MATHJAX_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js'
# init the App
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.LUX], external_scripts=[MATHJAX_CDN])

# -text
header_ode = html.H1('ODE solving exaples')
header_system = html.H1('ODE System solving example')
first_text = dcc.Markdown(r'''
                            The first initial value problem to be solved is:
                            $$
                            \begin{cases}
                            \frac{du}{dt} = u(t) \\
                            u(0) = 2 \\
                            \end{cases}                            
                            $$
                            The exact solution of this problem is given by: $u(t) = 2e^t$
                            ''',  mathjax=True)
second_text = dcc.Markdown(r'''
                           The first initial value problem to be solved is:
                            $$ 
                            \begin{cases}
                            \frac{du}{dt} = \frac{1}{u(t)} \\
                            u(0) = 1 \\
                            \end{cases}                             
                            $$
                            The exact solution of this problem is given by: $u(t) = \sqrt{2t + 1}$
                            ''', mathjax=True)

system1_text = dcc.Markdown(r'''
                           The first initial value problem to be solved is:
                            $$ 
                            \begin{cases}
                            \frac{du}{dt} = \alpha u - \beta uv \\
                            \frac{dv}{dt} = \delta uv - \gamma v \\
                            u(0) = 1.5, v(0) = 1 \\
                            \end{cases}                             
                            $$
                            $\alpha = \beta = \gamma = \delta = 1$
                            ''', mathjax=True)

system2_text = dcc.Markdown(r'''
                           The first initial value problem to be solved is:
                            $$ 
                            \begin{cases}
                            \frac{du}{dt} = u^2 + v + cos(t) - (1 + t^2 + sin(t)) \\
                            \frac{dv}{dt} = uv + 2t - (1 + t^2)sin(t) \\
                            u(0) = 0, v(0) = 1 \\
                            \end{cases}                             
                            $$
                            The exact solution of this problem is given by: $u(t) = sin(t), v = 1 + t^2$
                            ''', mathjax=True)
# calculation for ODE exaples
npr.seed(4155)
nx = 10
x = np.linspace(0, 1, nx)
num_hidden_neurons = [200, 100]
num_iter = 1000
lmb = 1e-3

set_EXAMPLE('first')
t1_start = process_time()
P1 = neural_network_solve_ode(x, num_hidden_neurons , num_iter , lmb)
t1_stop = process_time()
g_dnn_ag1 = g_trial_solution(x, P1)
g_analytical1 = g_exact(x)

tnum_start1 = process_time()
s1, t1 = implicit_euler(lambda t, s: s, tspan=np.array([0.0, 1.0]), y0=2, num_steps=nx)
tnum_stop1 = process_time()

set_EXAMPLE('second')
t2_start = process_time()
P2 = neural_network_solve_ode(x, num_hidden_neurons , num_iter , lmb)
t2_stop = process_time()
g_dnn_ag2 = g_trial_solution(x, P2)
g_analytical2 = g_exact(x)

tnum_start2 = process_time()
s2, t2 = implicit_euler(lambda t, s: 1 / s, tspan=np.array([0.0, 1.0]), y0=1, num_steps=nx)
tnum_stop2 = process_time()

# ODE figures
ode1 = go.Figure()
ode1.add_trace(go.Scatter(x=x, y=g_analytical1, mode='markers+lines', name='Exact u'))
ode1.add_trace(go.Scatter(x=x, y=g_dnn_ag1[0], mode='lines', name='ANN u'))
ode1.add_trace(go.Scatter(x=t1, y=s1.flatten(), mode='lines', name='Euler u'))
ode1.update_layout(title='Approximation', xaxis_title='t', yaxis_title='u', legend=dict(x=0, y=1))

ode2 = go.Figure()
ode2.add_trace(go.Scatter(x=x, y=g_analytical2, mode='markers+lines', name='Exact u'))
ode2.add_trace(go.Scatter(x=x, y=g_dnn_ag2[0], mode='lines', name='ANN u'))
ode2.add_trace(go.Scatter(x=t2, y=s2.flatten(), mode='lines', name='Euler u'))
ode2.update_layout(title='Approximation', xaxis_title='t', yaxis_title='u',legend=dict(x=0, y=1))

# ODE systems
ts1 = np.linspace(0, 12, 100)
prey_net, pred_net, prey_num, pred_num = get_system1(ts1)

system1 = go.Figure()
system1.add_trace(go.Scatter(x=ts1, y=prey_net, mode='lines', name='ANN u'))
system1.add_trace(go.Scatter(x=ts1, y=pred_net, mode='lines', name = 'ANN v'))
system1.add_trace(go.Scatter(x=ts1, y=prey_num, mode='markers', name = 'Numerical u'))
system1.add_trace(go.Scatter(x=ts1, y=pred_num, mode='markers', name = 'Numerical v'))
system1.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=1))

ts2 = np.linspace(0, 3, 50)
net_sol1, net_sol2 = get_system2(ts2)

system2 = go.Figure()
system2.add_trace(go.Scatter(x=ts2, y=net_sol1, mode='lines', name='ANN u'))
system2.add_trace(go.Scatter(x=ts2, y=net_sol2, mode='lines', name = 'ANN v'))
system2.add_trace(go.Scatter(x=ts2, y=np.sin(ts2), mode='markers', name = 'Numerical u'))
system2.add_trace(go.Scatter(x=ts2, y=1 + ts2**2, mode='markers', name = 'Numerical v'))
system2.update_layout(title='Approximation', xaxis_title='t', legend=dict(x=0, y=1))

#%%
tab1 = html.Div([
        dbc.Row([
            dbc.Col([header_ode])
        ]),
        dbc.Row([
            dbc.Col(
                [first_text,
                dcc.Graph(figure=ode1)], width={'size': 6, 'offset': 0,'order': 1}
            ),
            dbc.Col(
                [second_text,
                 dcc.Graph(figure=ode2)], width={'size': 6, 'offset': 0,'order': 2}
            ),
        ])
])

tab2 = html.Div([
        dbc.Row([
            dbc.Col([header_system])
        ]),
        dbc.Row([
            dbc.Col(
                [system1_text,
                dcc.Graph(figure=system1)], width={'size': 6, 'offset': 0,'order': 1}
            ),
            dbc.Col(
                [system2_text,
                 dcc.Graph(figure=system2)], width={'size': 6, 'offset': 0,'order': 2}
            ),
        ])
])

# layout
app.layout = html.Div(style={'padding': 25},
    children=
    [
        dbc.Tabs([
            dbc.Tab(tab1, label="Tab 1"),
            dbc.Tab(tab2, label='Tab 2')
        ])
    ]
)


# run the App
app.run_server(debug=True)