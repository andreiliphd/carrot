from dash.dependencies import Input, Output, State
from app import app
from carrotlayout import *
import carrotmysql


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'loss':
        return loss_display()
    elif tab == 'accuracy':
        return accuracy_display()
    elif tab == 'model':
        return model_display()
    elif tab == 'parameters':
        return parameters_display()
    elif tab == 'gradients':
        return gradients_display()


# Setting to 0 initial value of the slider
@app.callback(Output('parameters-slider', 'value'),
              [Input('parameters-dropdown', 'value')])
def update_output(input1):
    return 0



@app.callback(Output('parameters-graph', 'figure'),
              [Input('parameters-slider', 'value')],
               [State('parameters-dropdown', 'value')])
def update_output(input1, input2):

    arr = carrotmysql.query.batch_seq(input1)
    return {
        'data': [
            {'x': arr[0],
             'y': arr[1],
             'z': arr[2],
             'type': 'scatter3d', 'name': input2}
        ],
        'layout': {
            'title': input2
        }
    }

@app.callback(Output('parameters-slider', 'max'),
              [Input('parameters-dropdown', 'value')])
def update_output(input1):
    carrotmysql.query.search_layer(input1, parameter=True)
    return carrotmysql.query.batch_size







# Setting to 0 initial value of the slider
@app.callback(Output('gradients-slider', 'value'),
              [Input('gradients-dropdown', 'value')])
def update_output(input1):
    return 0



@app.callback(Output('gradients-graph', 'figure'),
              [Input('gradients-slider', 'value')],
               [State('gradients-dropdown', 'value')])
def update_output(input1, input2):

    arr = carrotmysql.query.batch_seq(input1)
    return {
        'data': [
            {'x': arr[0],
             'y': arr[1],
             'z': arr[2],
             'type': 'scatter3d', 'name': input2}
        ],
        'layout': {
            'title': input2
        }
    }

@app.callback(Output('gradients-slider', 'max'),
              [Input('gradients-dropdown', 'value')])
def update_output(input1):
    carrotmysql.query.search_layer(input1, parameter=False)
    return carrotmysql.query.batch_size