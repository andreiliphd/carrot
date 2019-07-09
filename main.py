import carrotdisplay
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='loss', children=[
        dcc.Tab(label='Loss', value='loss'),
        dcc.Tab(label='Accuracy', value='accuracy'),
        dcc.Tab(label='Model', value='model'),
        dcc.Tab(label='Parameters', value='parameters'),
        dcc.Tab(label='Gradients', value='gradients'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'loss':
        return carrotdisplay.loss_display()
    elif tab == 'accuracy':
        return carrotdisplay.accuracy_display()
    elif tab == 'model':
        return carrotdisplay.model_display()
    elif tab == 'parameters':
        return carrotdisplay.parameters_display()
    elif tab == 'gradients':
        return carrotdisplay.gradients_display()


if __name__ == '__main__':
    app.run_server(debug=True)
