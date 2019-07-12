import dash_html_components as html
import dash_core_components as dcc
import carrotmysql

def loss_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='loss-graph',
                figure={
                    'data': [
                        {'x': carrotmysql.query.epoch, 'y': carrotmysql.query.training_loss, 'type': 'line',
                         'name': 'Train loss'},
                        {'x': carrotmysql.query.epoch, 'y': carrotmysql.query.test_loss, 'line': 'bar',
                         'name': 'Validation loss'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )
        ]
        )

    ]
    )
    return display

def accuracy_display():
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='accuracy-graph',
                figure={
                    'data': [
                        {'x': carrotmysql.query.epoch, 'y': carrotmysql.query.training_accuracy,
                         'type': 'bar',
                         'name': 'Train loss'},
                        {'x': carrotmysql.query.epoch, 'y': carrotmysql.query.test_accuracy,
                         'type': 'bar',
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


def parameters_display():
    carrotmysql.query.search_layer(carrotmysql.query.current_layer, parameter=True)
    arr = carrotmysql.query.batch_seq(0)
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='parameters-graph',
                figure={
                    'data': [
                        {'x': arr[0],
                         'y': arr[1],
                         'z': arr[2],
                         'type': 'scatter3d', 'name': carrotmysql.query.current_layer}
                    ],
                    'layout': {
                        'title': carrotmysql.query.current_layer
                    }
                }
            ),
            dcc.Slider(
                id='parameters-slider',
                min=-0,
                max=carrotmysql.query.batch_size,
                step=1,
                value=0
            ),
            dcc.Dropdown(
                id='parameters-dropdown',
                options=carrotmysql.query.get_options(),
                value=carrotmysql.query.current_layer
            )

        ]
        )
    ])
    return display

def gradients_display():
    carrotmysql.query.search_layer(carrotmysql.query.current_layer, parameter=False)
    arr = carrotmysql.query.batch_seq(0)
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='gradients-graph',
                figure={
                    'data': [
                        {'x': arr[0],
                         'y': arr[1],
                         'z': arr[2],
                         'type': 'scatter3d', 'name': carrotmysql.query.current_layer}
                    ],
                    'layout': {
                        'title': carrotmysql.query.current_layer
                    }
                }
            ),
            dcc.Slider(
                id='gradients-slider',
                min=-0,
                max=carrotmysql.query.batch_size,
                step=1,
                value=0
            ),
            dcc.Dropdown(
                id='gradients-dropdown',
                options=carrotmysql.query.get_options(),
                value=carrotmysql.query.current_layer
            )

        ]
        )
    ])
    return display
    return display


