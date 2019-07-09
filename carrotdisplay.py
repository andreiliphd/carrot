import carrotmongo
import dash_core_components as dcc
import dash_html_components as html



def loss_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': carrotmongo.query.epoch, 'y': carrotmongo.query.train_loss, 'type': 'bar',
                         'name': 'Train loss'},
                        {'x': carrotmongo.query.epoch, 'y': carrotmongo.query.test_loss, 'type': 'bar',
                         'name': 'Validation loss'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )


        ]

        )

    ])
    return display

def accuracy_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': carrotmongo.query.epoch, 'y': carrotmongo.query.train_accuracy, 'type': 'bar',
                         'name': 'Train loss'},
                        {'x': carrotmongo.query.epoch, 'y': carrotmongo.query.test_accuracy, 'type': 'bar',
                         'name': 'Validation loss'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )


        ]

        )

    ])
    return display


def model_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )
        ])
    ])
    return display

def parameters_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': carrotmongo.query.search_layer('c1.weight', parameter=False)[0],
                         'y': carrotmongo.query.search_layer('c1.weight', parameter=False)[1],
                         'z': carrotmongo.query.search_layer('c1.weight', parameter=False)[2],
                         'type': 'scatter3d', 'name': 'c1.weight'}
                    ],
                    'layout': {
                        'title': 'c1.weight parameter'
                    }
                }
            )
        ])
    ])
    return display


def gradients_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': carrotmongo.query.search_layer('c1.weight', parameter=True)[0],
                         'y': carrotmongo.query.search_layer('c1.weight', parameter=True)[1],
                         'z': carrotmongo.query.search_layer('c1.weight', parameter=True)[2],
                         'type': 'scatter3d', 'name': 'c1.weight'}
                    ],
                    'layout': {
                        'title': 'c1.weight gradient'
                    }
                }
            )
        ])
    ])
    return display
    return display