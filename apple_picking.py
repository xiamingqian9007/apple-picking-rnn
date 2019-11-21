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

class ApplePicking:
    def __init__(self, data_dir="/home/hmt_user/ROB_537_LBC/project/all_grasp_data/", model_dir='./Models/', test_split=0.05, window_size=10):
        self.data_dir = data_dir
        self. model_dir = model_dir
        self.window_size = window_size
        self.train, self.test = self.load_data(test_split)

    def add_force_mag_data(self, row):
        fx = row['/wrench.fx']
        fy = row['/wrench.fy']
        fz = row['/wrench.fz']
        f_mag = math.sqrt(fx**2 + fy**2 + fz**2)
        return f_mag

    def format_data(self, df):
        input_col = ['force_mag', '/manipulator_pose.x', '/manipulator_pose.y', '/manipulator_pose.z', '/manipulator_pose.rx', '/manipulator_pose.ry', '/manipulator_pose.rz', '/manipulator_pose.rw']
        output_col = ['/wrench.fx', '/wrench.fy', '/wrench.fz']
        n_df = len(df) - self.window_size
        X = []
        Y = []
        for i in range(0, n_df):
            seq_X = df.iloc[i: i+self.window_size]
            seq_Y = seq_X.iloc[-1]
            x = seq_X[input_col].values
            y = seq_Y[output_col].values
            X.append(x)
            Y.append(y)
        return X, Y

    def get_data(self, files_list):
        X_data = []
        Y_data = []
        for file in files_list:
            temp_df = pd.read_csv(file)

            temp_df = temp_df - temp_df.iloc[0].values.squeeze()
            temp_df = temp_df.loc[1:]
            temp_df['force_mag'] = temp_df.apply(lambda row: self.add_force_mag_data(row), axis=1)

            temp_X, temp_Y = self.format_data(temp_df)
            X_data = X_data + temp_X
            Y_data = Y_data + temp_Y

        X_data = np.array(X_data, dtype=np.float64)
        Y_data = np.array(Y_data, dtype=np.float64)
        return X_data, Y_data

    def load_data(self, test_split):
        print ("Loading Data ...")
        all_files = [file for file in glob.glob(self.data_dir + "*.csv", recursive=False)]
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
        new_model_dir = self.model_dir + model_name + '/'
        if not os.path.exists(new_model_dir):
            os.makedirs(new_model_dir)

        # clear tmp directory to save disk space
        logdir = "/tmp/tflearn_logs/"
        if clear_tmp == True and os.path.exists(logdir): 
            os.system("rm -r " + logdir + "*")

        model_path = new_model_dir + model_name + '.h5'
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

        if is_restore == False:
            self.model = self.perf_nnet_conv1D(feature_dim, model_path)
        else:
            self.model = keras.models.load_model(model_path)

        (train_inputs, train_labels) = self.train
        (test_inputs, test_labels) = self.test
        self.model.fit(train_inputs, train_labels, epochs=n_epoch, verbose=1, callbacks=[tensorboard_cb, checkpoint_cb], validation_data=(test_inputs, test_labels), shuffle=True)

def main():
    apple_picking_obj = ApplePicking()
    apple_picking_obj.train_network()
if __name__ == '__main__':
    main()