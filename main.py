import dash_html_components as html
import dash_core_components as dcc
from app import app
import carrotcallbacks



app.layout = html.Div([
    dcc.Tabs(id="tabs", value='loss', children=[
        dcc.Tab(label='Loss', value='loss'),
        dcc.Tab(label='Accuracy', value='accuracy'),
        dcc.Tab(label='Parameters', value='parameters'),
        dcc.Tab(label='Gradients', value='gradients'),
    ]),
    html.Div(id='tabs-content')
])



if __name__ == '__main__':
    app.run_server(debug=True)
