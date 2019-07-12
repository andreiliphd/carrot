import mongoengine as mo
import datetime
import pickle
import os
import glob
import numpy as np

try:
    from configcarrot import *
except ImportError:
    login = '' # Please provide your login to MongoDB
    password = '' # Please provide your password to MongoDB
    host = '' # Please provide your host name to MongoDB


class Carrot(mo.Document):
    date_time = mo.DateTimeField(required=True)
    epoch = mo.IntField()
    train_loss = mo.FloatField()
    test_loss = mo.FloatField()
    train_accuracy = mo.FloatField()
    test_accuracy = mo.FloatField()
    parameters = mo.FileField()
    gradients = mo.FileField()


class Query():
    def __init__(self):
        self.model = Carrot
        self.keys = []
        self.date_time = []
        self.epoch = []
        self.train_loss = []
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        if not os.path.exists(os.path.join('carrot', 'preprocessing')):
            mo.connect(host='mongodb+srv://' + login + ':' + password + '@' + host,
                       authentication_source='admin')
            self.init_populate()
        self.get_keys()
        if not os.path.exists(os.path.join('carrot', 'processed')):
            self.processing()
        self.populate_training()
        self.current_option = ''
        self.get_options()




    def init_populate(self):
        print('init_populate started')
        for selected_data in self.model.objects(epoch=1):
            data_layer = selected_data.parameters.read()
            data_layer = pickle.loads(data_layer)
            self.keys = list(data_layer.keys())

        result = []
        date_time = []
        epoch = []
        train_loss = []
        test_loss = []
        train_accuracy = []
        test_accuracy = []
        for obj_num, selected_data in enumerate(self.model.objects):
            print('Started loop')
            date_time.append(selected_data.date_time)
            epoch.append(selected_data.epoch)
            train_loss.append(selected_data.train_loss)
            test_loss.append(selected_data.test_loss)
            train_accuracy.append(selected_data.train_accuracy)
            test_accuracy.append(selected_data.test_accuracy)

            parameters = pickle.loads(selected_data.parameters.read())
            gradients = pickle.loads(selected_data.gradients.read())
            os.makedirs(os.path.join('carrot','preprocessing'), exist_ok=True)
            for number, key in enumerate(self.keys):
                print(key)
                print(type(key))
                print(parameters[key])

                pickle.dump(parameters[key], open(
                    os.path.join('carrot', 'preprocessing', 'parameters_' +
                                 key + '_' + str(obj_num) + '_' + str(number)), "wb"))

                pickle.dump(gradients[key], open(
                    os.path.join('carrot', 'preprocessing','gradients_' +
                                 key + '_' + str(obj_num) + '_' + str(number)), "wb"))
        print('finishing')
        result.append(date_time)
        result.append(epoch),
        result.append(train_loss)
        result.append(test_loss)
        result.append(train_accuracy)
        result.append(test_accuracy)
        pickle.dump(result, open(
            os.path.join('carrot', 'preprocessing', 'data'), "wb"))

    def get_keys(self):
        os.makedirs(os.path.join('carrot', 'processed'), exist_ok=True)
        files = glob.glob('carrot/preprocessing/*')
        keys = []
        for file in files:
            file = file[21:]
            if 'data' in file:
                continue
            name = file.split('_')
            if name[1] not in keys:
                keys.append(name[1])
        self.keys = keys

    def processing(self):
        os.makedirs(os.path.join('carrot', 'processed'), exist_ok=True)
        files = glob.glob('carrot/preprocessing/*')
        maximum = []
        for file in files:
            file = file[21:]
            if 'data' in file:
               continue
            name = file.split('_')
            maximum.append(int(name[2]))
        maximum = max(maximum)

        for num, key in enumerate(self.keys):
            results_param = []
            results_grad = []
            for seq in range(18):
                for file in files:
                    file_full = file
                    file = file[21:]
                    if 'data' in file:
                        continue
                    name = file.split('_')
                    if name[1] == key and name[0] == "parameters" and name[2] == str(seq):
                        _ = pickle.load(open(file_full, 'rb'))
                        results_param.append(_)
                    if name[1] == key and name[0] == "gradients" and name[2] == str(seq):
                        _ = pickle.load(open(file_full, 'rb'))
                        results_grad.append(_)
            pickle.dump(results_param, open(
                    os.path.join('carrot', 'processed', 'parameters_' +
                                 key), "wb"))
            pickle.dump(results_grad, open(
                    os.path.join('carrot', 'processed', 'gradients_' +
                                 key), "wb"))
    def populate_training(self):
        os.makedirs(os.path.join('carrot', 'processed'), exist_ok=True)
        files = glob.glob('carrot/preprocessing/*')
        for file in files:
            full_file = file
            file = file[21:]
            if 'data' in file:
                _ = pickle.load(open(full_file, 'rb'))
                self.date_time = _[0]
                self.epoch = _[1]
                self.train_loss = _[2]
                self.test_loss = _[3]
                self.train_accuracy = _[4]
                self.test_accuracy = _[5]

    def search_layer(self, layer, parameter = True):
        os.makedirs(os.path.join('carrot', 'processed'), exist_ok=True)
        files = glob.glob('carrot/processed/*')
        result = []
        for file in files:
            full_file = file
            file = file[17:]
            name = file.split('_')
            if parameter:
                if (name[1] == layer) and (name[0] == 'parameters'):
                    result = pickle.load(open(full_file, 'rb'))
                    break
            else:
                if (name[1] == layer) and (name[0] == 'gradients'):
                    result = pickle.load(open(full_file, 'rb'))
                    break

        return result

    def shape_processing(self, layer, parameter=True, seq=0):
        self.current_option = layer
        arr = self.search_layer(layer, parameter)
        arr = np.concatenate(arr, axis=0)

        if len(arr.shape) != 4:
            try:
                arr = arr.reshape(-1, 3, 3, 3)
            except Exception:
                try:
                    arr = arr.reshape(-1, 2, 2, 2)
                except Exception:
                    try:
                        arr = arr.reshape(-1, 1, 1, 1)
                    except Exception:
                        pass
        print('Shape of the array', arr.shape)
        batch_size = arr.shape[0] - 1
        x = arr[seq].transpose((0, 1, 2)).ravel()
        y = arr[seq].transpose((1, 2, 0)).ravel()
        z = arr[seq].transpose((2, 0, 1)).ravel()
        return (batch_size, x, y, z)

    def get_options(self):
        # options = [
        #     {'label': 'New York City', 'value': 'NYC'},
        #     {'label': 'Montréal', 'value': 'MTL'},
        #     {'label': 'San Francisco', 'value': 'SF'}
        # ]
        options = []
        for num, key in enumerate(self.keys):
            if num == 0:
                self.current_option = key
            dic = {}
            dic['label'] = key
            dic['value'] = key
            options.append(dic)
        return options

    def print_shape(self):
        for key in self.keys:
            arr = self.search_layer(key, parameter=False)
            print(arr[0].shape)


query = Query()

