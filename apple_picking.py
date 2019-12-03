import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, GlobalAveragePooling1D, LSTM
from sklearn.utils import shuffle

try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
import numpy as np
import pandas as pd 
PANDAS_VERSION = int(pd.__version__.split('.')[1])


import math
import matplotlib
import matplotlib.pyplot as plt

import os
import sys

ROOT = os.path.dirname(os.path.realpath(__file__))

class ApplePicking:

    def __init__(self, network_type, window_size, smoothing_window = 1, use_force=True,
                 train_dir='training_data', test_dir='testing_data', model_folder='models'):


        self.model = None

        self.network_type = network_type
        self.train_dir = os.path.join(ROOT, train_dir)
        self.model_dir = os.path.join(ROOT, model_folder)
        self.test_dir = os.path.join(ROOT, test_dir)
        self.model_name = '{}_ws{}_smooth{}{}.h5'.format(network_type, window_size, smoothing_window, '_noforce' if not use_force else '')

        self.window_size = window_size
        self.smoothing_window = smoothing_window

        joint_cols = ['/joint_states.shoulder_lift_joint', '/joint_states.elbow_joint', '/joint_states.wrist_1_joint',
                      '/joint_states.wrist_2_joint', '/joint_states.wrist_3_joint']
        force_cols = ['/manipulator_wrench.fx', '/manipulator_wrench.fy', '/manipulator_wrench.fz',
                      '/manipulator_wrench.tx', '/manipulator_wrench.ty', '/manipulator_wrench.tz']

        if use_force:
            self.INPUT_COLS = force_cols + joint_cols
        else:
            self.INPUT_COLS = joint_cols

        self.OUTPUT_COLS = ['/ground_truth.x', '/ground_truth.y', '/ground_truth.z']

        self.train = None
        self.validation = None
        self.train_files = None
        self.validation_files = None

    @property
    def model_path(self):
        return os.path.join(self.model_dir, self.model_name)

    def load_from_cache(self, load_data=False):
        self.model = keras.models.load_model(self.model_path)
        # Metadata for testing/validation data
        with open(self.model_path.replace('.h5', '.pickle'), 'rb') as fh:
            metadata = pickle.load(fh)

        self.train_files = [os.path.join(self.train_dir, os.path.split(path)[-1]) for path in metadata['train']]
        self.validation_files = [os.path.join(self.train_dir, os.path.split(path)[-1]) for path in metadata['validation']]
        if load_data:
            self.train, self.validation = self.load_all_data()


    def add_force_mag_data(self, row):
        fx = row['/wrench.fx']
        fy = row['/wrench.fy']
        fz = row['/wrench.fz']
        f_mag = math.sqrt(fx**2 + fy**2 + fz**2)
        return f_mag

    def adjust_force_data(self, df):
        force_cols = ['/manipulator_wrench.fx', '/manipulator_wrench.fy', '/manipulator_wrench.fz', '/manipulator_wrench.tx', '/manipulator_wrench.ty', '/manipulator_wrench.tz']
        subset = df[force_cols]
        tared = subset - subset.iloc[0]
        df[force_cols] = tared
        # df['force_mag'] = (tared ** 2).sum(axis=1) ** 0.5
        df = df.iloc[1:].reset_index(drop=True)
        return df

    def format_data(self, df):
        n_df = len(df) - self.window_size
        X = []
        Y = []
        for i in range(0, n_df):
            seq_X = df.iloc[i: i+self.window_size]
            seq_Y = seq_X.iloc[-1]
            x = seq_X[self.INPUT_COLS].values
            y = seq_Y[self.OUTPUT_COLS].values
            X.append(x)
            Y.append(y)
        return X, Y

    def smooth_data(self, df):
        cols_to_smooth = [col for col in self.INPUT_COLS if col.startswith('/manipulator_wrench')]
        if PANDAS_VERSION < 18:
            df[cols_to_smooth] = pd.rolling_mean(df[cols_to_smooth], self.smoothing_window, self.smoothing_window)
        else:
            df[cols_to_smooth] = df[cols_to_smooth].rolling(self.smoothing_window).mean()
        df = df.dropna().reset_index(drop=True)
        return df

    def get_data(self, files_list):
        X_data = []
        Y_data = []
        mode_col = '/mode./mode'
        for file in files_list:
            temp_df = pd.read_csv(file)
            temp_df = self.adjust_force_data(temp_df)
            
            # Taking only the mode 3 columns
            temp_df = temp_df.loc[temp_df[mode_col] == 3]
            temp_df = temp_df.dropna().reset_index(drop=True)

            if self.smoothing_window > 1:
                temp_df = self.smooth_data(temp_df) # Smoothing using moving average filter

            temp_X, temp_Y = self.format_data(temp_df)
            X_data = X_data + temp_X
            Y_data = Y_data + temp_Y

        X_data = np.array(X_data, dtype=np.float64)
        Y_data = np.array(Y_data, dtype=np.float64)
        return X_data, Y_data

    def load_all_data(self, val_split=None):
        print ("Loading Data ...")

        if not self.train_files or not self.validation_files:
            all_files = [file for file in glob.glob(os.path.join(self.train_dir, "*.csv"))]
            random.shuffle(all_files)

            split_idx = int(round(val_split*len(all_files)))

            self.train_files = all_files[0: split_idx]
            self.validation_files = all_files[split_idx: ]
        
        train = self.get_data(self.train_files)
        validation = self.get_data(self.validation_files)

        return train, validation

    def load_test_data(self):

        test_folder = os.path.join(ROOT, self.test_dir)
        test_files = [file for file in glob.glob(os.path.join(test_folder, "*.csv"))]
        test_X, test_Y = self.get_data(test_files)
        return test_X, test_Y

    def perf_nnet_conv1D(self, feature_dim, output_dim, model_path):
        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu', input_shape=(self.window_size, feature_dim)))
        model.add(Conv1D(64, 3, activation='relu'))
        # model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        model.add(Conv1D(128, 3, activation='relu'))
        # model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        # model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mae', optimizer="adam", metrics=['mae', 'accuracy'])
        print(model.summary())
        return model

    def perf_nnet_LSTM(self, feature_dim, output_dim, model_path):
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(self.window_size, feature_dim)))
        # model.add(LSTM(hidden_neurons, return_sequences=True))
        model.add(LSTM(32))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mae', optimizer="adam", metrics=['mae', 'accuracy'])
        print(model.summary())
        model.save(model_path)
        return model

    def perf_nnet_ann(self, feature_dim, output_dim, model_path):
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(feature_dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss='mae', optimizer="adam", metrics=['mae', 'accuracy'])
        print(model.summary())
        return model

    def train_network(self, val_split, n_epoch=1000, clear_tmp=True):
        feature_dim = len(self.INPUT_COLS)
        output_dim = len(self.OUTPUT_COLS)
        self.train, self.validation = self.load_all_data(val_split)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # clear tmp directory to save disk space
        logdir = "/tmp/tflearn_logs/"
        if clear_tmp == True and os.path.exists(logdir): 
            os.system("rm -r " + logdir + "*")

        model_path = self.model_path

        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

        (train_inputs, train_labels) = self.train
        (val_inputs, val_labels) = self.validation

        print (train_inputs.shape, train_labels.shape)
        print (val_inputs.shape, val_labels.shape)

        if self.network_type == 'ANN':
            train_inputs = train_inputs.reshape(-1, feature_dim)
            val_inputs = val_inputs.reshape(-1, feature_dim)
            print (train_inputs.shape, val_inputs.shape)
            self.model = self.perf_nnet_ann(feature_dim, output_dim, model_path)
            shuffle = True

        elif self.network_type == 'Conv1D':
            self.model = self.perf_nnet_conv1D(feature_dim, output_dim, model_path)
            shuffle = True

        elif self.network_type == 'LSTM':
            self.model = self.perf_nnet_LSTM(feature_dim, output_dim, model_path)
            shuffle = False

        else:
            raise NotImplementedError('Currently no implementation available for model {}'.format(self.network_type))

        self.model.fit(train_inputs, train_labels, epochs=n_epoch, verbose=1, callbacks=[tensorboard_cb, checkpoint_cb], validation_data=(val_inputs, val_labels), shuffle=shuffle)
        self.model.save(model_path)
        metadata = {
            'validation': self.validation_files,
            'train': self.train_files
        }
        with open(model_path.replace('.h5', '.pickle'), 'wb') as fh:
            pickle.dump(metadata, fh)

    def predict_network(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions

    def orientation_error(self, targets, predictions):
        orientation_error = []
        for i in range(len(predictions)):
            vec_1 = predictions[i]
            vec_1 = vec_1/np.linalg.norm(vec_1)

            vec_2 = targets[i]
            vec_2 = vec_2/np.linalg.norm(vec_2)
            radian = np.arccos(np.clip(np.dot(vec_1, vec_2), -1.0, 1.0))
            theta = (180.0/math.pi)*radian
            orientation_error.append(theta)
        print(np.shape(orientation_error))
        return orientation_error

if __name__ == '__main__':

    # Call, e.g. python apple_picking.py train ANN 3

    mode = 'train'
    network = 'ANN'      # ANN, Conv1D, LSTM
    smooth = 1
    window_size = 5

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if len(sys.argv) > 2:
        network = sys.argv[2]

    if len(sys.argv) > 3:
        smooth = int(sys.argv[3])

    if network == 'ANN':
        window_size = 1
        
    apple_model = ApplePicking(network, window_size=window_size, smoothing_window=smooth)

    if mode == 'train':
        apple_model.train_network(0.10, n_epoch=1000)

    elif mode == 'predict':

        apple_model.load_from_cache(load_data=True)

        output_base = os.path.join(ROOT, 'output_{}_{}.png')



        # Plotting the error in the orientation
        plt.clf()
        labels = []
        errors = []
        for label, dataset in [['Training', apple_model.train],
                               ['Validation', apple_model.validation],
                               ['Testing', apple_model.load_test_data()]]:

            inputs, outputs = dataset
            inputs, outputs = shuffle(inputs, outputs, random_state=0)
            predictions = apple_model.predict_network(inputs)
            orientation_error = apple_model.orientation_error(outputs, predictions)
            orientation_arr = np.array(orientation_error)

            labels.append(label)
            errors.append(orientation_error)

        plt.boxplot(errors, labels=labels)

        plt.title('Orientation Error in degrees', fontsize=35)
        plt.xlabel("Time Steps (0.1s)", fontsize=25)
        plt.ylabel("Orrientation Error (deg)", fontsize=25)
        plt.show()
        # plt.savefig(output_base.format(data_label.lower(), label.lower()))
        
        # Plotting individual X,Y,Z components
        for idx, label in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            plt.clf()
            plt.plot(outputs[:, idx], label=label, color='r')
            plt.plot(predictions[:, idx], label='{} (pred)'.format(label), color='g')
            plt.legend()
            plt.title('{} Data - {} coord'.format(data_label, label))
            # plt.savefig(output_base.format(data_label.lower(), label.lower()))
            # plt.show()

        # for data_label, dataset in [['Training', random.choice(apple_picking_obj.train)], ['Validation', random.choice(apple_picking_obj.test)]]:
        #     # Test a data set
        #     inputs, outputs = apple_picking_obj.train
        #     predictions = apple_picking_obj.predict_network(model_name, inputs)

        #     for idx, label in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        #         plt.clf()
        #         plt.plot(outputs[:, idx], label=label, color='r')
        #         plt.plot(predictions[:, idx], label='{} (pred)'.format(label), color='g')
        #         plt.legend()
        #         plt.title('{} Data - {} coord'.format(data_label, label))
        #         plt.show()
        #         plt.savefig(output_base.format(data_label.lower(), label.lower()))
