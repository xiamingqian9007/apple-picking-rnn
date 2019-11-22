import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, GlobalAveragePooling1D, LSTM

import random
import numpy as np
import pandas as pd 

import math
import matplotlib
import matplotlib.pyplot as plt

import os

ROOT = os.path.dirname(os.path.realpath(__file__))

class ApplePicking:

    INPUT_COLS = ['force_mag', '/manipulator_pose.x', '/manipulator_pose.y', '/manipulator_pose.z', '/manipulator_pose.rx', '/manipulator_pose.ry', '/manipulator_pose.rz', '/manipulator_pose.rw']
    OUTPUT_COLS = ['/wrench.fx', '/wrench.fy', '/wrench.fz']

    def __init__(self, data_folder = 'all_grasp_data', model_folder = 'models', test_split=0.05, window_size=10):
        self.data_dir = os.path.join(ROOT, data_folder)
        self.model_dir = os.path.join(ROOT, model_folder)
        self.window_size = window_size
        self.train, self.test = self.load_all_data(test_split)


    def add_force_mag_data(self, row):
        fx = row['/wrench.fx']
        fy = row['/wrench.fy']
        fz = row['/wrench.fz']
        f_mag = math.sqrt(fx**2 + fy**2 + fz**2)
        return f_mag

    def adjust_force_data(self, df):
        force_cols = ['/wrench.fx', '/wrench.fy', '/wrench.fz']
        subset = df[force_cols]
        tared = subset - subset.iloc[0]
        df[force_cols] = tared
        df['force_mag'] = (tared ** 2).sum(axis=1) ** 0.5

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

    def get_data(self, files_list):
        X_data = []
        Y_data = []

        print('Loading data...')
        for file in files_list:
            temp_df = pd.read_csv(file)
            self.adjust_force_data(temp_df)


            temp_X, temp_Y = self.format_data(temp_df)
            X_data = X_data + temp_X
            Y_data = Y_data + temp_Y

        X_data = np.array(X_data, dtype=np.float64)
        Y_data = np.array(Y_data, dtype=np.float64)
        return X_data, Y_data

    def load_all_data(self, test_split):
        print ("Loading Data ...")

        all_files = [file for file in glob.glob(os.path.join(self.data_dir, "*.csv"), recursive=False)]
        random.shuffle(all_files)

        split_idx = int(round(test_split*len(all_files)))
        
        test_files = all_files[0: split_idx]
        train_files = all_files[split_idx: ]
        
        train_X, train_Y = self.get_data(train_files)
        test_X, test_Y = self.get_data(test_files)

        print (train_X.shape, train_Y.shape)
        print (test_X.shape, test_Y.shape)
        return (train_X, train_Y), (test_X, test_Y)

    def perf_nnet_conv1D(self, feature_dim, model_path):
        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu', input_shape=(self.window_size, feature_dim)))
        model.add(Conv1D(64, 3, activation='relu'))
        # model.add(MaxPooling1D(2))
        model.add(Dropout(0.3))

        model.add(Conv1D(128, 3, activation='relu'))
        # model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        # model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(3, activation='linear'))
        model.compile(loss='mae', optimizer="adam", metrics=['mae', 'accuracy'])
        print(model.summary())
        model.save(model_path)
        return model

    def train_network(self, n_epoch=1000, is_restore=False, clear_tmp=True):
        input_col = ['force_mag', '/manipulator_pose.x', '/manipulator_pose.y', '/manipulator_pose.z', '/manipulator_pose.rx', '/manipulator_pose.ry', '/manipulator_pose.rz', '/manipulator_pose.rw']
        feature_dim = len(input_col)

        model_name = 'force_vec_pred_ws' + str(self.window_size) + '_featdim' + str(feature_dim) # + '_e' + str(n_epoch)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # clear tmp directory to save disk space
        logdir = "/tmp/tflearn_logs/"
        if clear_tmp == True and os.path.exists(logdir): 
            os.system("rm -r " + logdir + "*")

        model_path = os.path.join(self.model_dir, '{}.h5'.format(model_name))
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

        if is_restore == False:
            self.model = self.perf_nnet_conv1D(feature_dim, model_path)
        else:
            self.model = keras.models.load_model(model_path)
            return

        (train_inputs, train_labels) = self.train
        (test_inputs, test_labels) = self.test
        self.model.fit(train_inputs, train_labels, epochs=n_epoch, verbose=1, callbacks=[tensorboard_cb, checkpoint_cb], validation_data=(test_inputs, test_labels), shuffle=True)



if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    mode = 'train'
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    apple_picking_obj = ApplePicking()

    if mode == 'train':

        apple_picking_obj.train_network(is_restore=False)

    elif mode == 'validate':
        apple_picking_obj.train_network(is_restore=True)
        output_base = os.path.join(ROOT, 'output_{}_{}.png')

        for data_label, dataset in [['Training', random.choice(apple_picking_obj.train)], ['Validation', random.choice(apple_picking_obj.test)]]:
            # Test a data set
            inputs, outputs = apple_picking_obj.train
            predictions = apple_picking_obj.model.predict(inputs)


            for idx, label in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
                plt.clf()
                plt.plot(outputs[:, idx], label=label, color='r')
                plt.plot(predictions[:, idx], label='{} (pred)'.format(label), color='g')
                plt.legend()
                plt.title('{} Data - {} coord'.format(data_label, label))
                plt.savefig(output_base.format(data_label.lower(), label.lower()))
