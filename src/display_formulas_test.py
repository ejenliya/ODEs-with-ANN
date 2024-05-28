import dash
from dash import dcc, html
from textwrap import dedent

# v2
MATHJAX_CDN = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML"

# v3
MATHJAX_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js'

app = dash.Dash(__name__, external_scripts=[MATHJAX_CDN])
# Define the layout of the app
app.layout = html.Div([
    html.H1("Formulas in Dash"),
    html.P("Here are some example formulas:"),
    html.P("Formula 1: "), dcc.Markdown('''$x^2 + y^2 = z^2$''',  mathjax=True),
    html.P("Formula 2: "), dcc.Markdown('$\\frac{d}{dx}(e^{ax}) = a \\cdot e^{ax}$'),
    html.P("Formula 3: "), dcc.Markdown('$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$', mathjax=True),
])


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)