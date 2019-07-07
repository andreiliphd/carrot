import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import datetime
import mongoengine
import math
import pickle

date_time = []
epoch = []
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
gradients = []

init_date = datetime.datetime.strptime('26 Sep 2012', '%d %b %Y')

def query_data():
    global init_date
    filtered_data = TrainingData.objects(date_time__gt = init_date)
    init_date = datetime.datetime.now()

    for selected_data in filtered_data:
        date_time.append(selected_data.date_time)
        epoch.append(selected_data.epoch)
        train_loss.append(selected_data.train_loss)
        test_loss.append(selected_data.test_loss)
        train_accuracy.append(selected_data.train_accuracy)
        test_accuracy.append(selected_data.test_accuracy)
        gradients.append(selected_data.gradients)


class TrainingData(mongoengine.Document):
    date_time = mongoengine.DateTimeField(required=True)
    epoch = mongoengine.IntField()
    train_loss = mongoengine.FloatField()
    test_loss = mongoengine.FloatField()
    train_accuracy = mongoengine.FloatField()
    test_accuracy = mongoengine.FloatField()
    gradients = mongoengine.BinaryField()


mongoengine.connect('pytorchboard')




def loss_display():
    query_data()
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': epoch, 'y': [math.log(n) for n in train_loss], 'type': 'bar', 'name': 'Train loss'},
                        {'x': epoch, 'y': [math.log(n) for n in train_loss], 'type': 'bar', 'name': 'Validation loss'}
                    ],
                    'layout': {
                        'title': 'Dash Data Visualization'
                    }
                }
            )
        ])
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

def gradients_display():
    query_data()
    array = pickle.loads(gradients[0])
    print(array.keys())
    experiment = array['c1.weight'][0]
    print(array['c2.weight'].shape)
    print(array['c3.weight'].shape)
    print(array['fc1.weight'].shape)
    print(array['fc2.weight'].shape)
    print(array['fc3.weight'].shape)
    x = experiment.transpose((0, 1, 2)).ravel()
    print(x.shape)
    y = experiment.transpose((1, 2, 0)).ravel()
    print(y.shape)
    z = experiment.transpose((2, 0, 1)).ravel()
    print(z.shape)
    display = html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='example-graph',
                figure={
                    'data': [
                        {'x': x, 'y': y, 'z': z, 'type': 'scatter3d', 'name': 'c1.weight'}
                    ],
                    'layout': {
                        'title': 'c1.weight gradient'
                    }
                }
            )
        ])
    ])
    return display

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='loss', children=[
        dcc.Tab(label='Loss', value='loss'),
        dcc.Tab(label='Model', value='model'),
        dcc.Tab(label='Gradients', value='gradients'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'loss':
        return loss_display()
    elif tab == 'model':
        return model_display()

    elif tab == 'gradients':
        return gradients_display()


if __name__ == '__main__':
    app.run_server(debug=True)
