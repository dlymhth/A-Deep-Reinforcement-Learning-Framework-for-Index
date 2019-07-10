# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:47:12 2019

@author: dlymhth
"""
import os
import random

import pandas as pd
import numpy as np

class Dataset:
    '''Generate dataset.
    '''
    def __init__(self, parameters):
        self.data_file_location = parameters.data_file_location
        self.varieties = parameters.varieties
        self.features = parameters.features

        self.train_data_rate = parameters.train_data_rate

        self.n_timesteps = parameters.n_timesteps
        self.n_batch = parameters.n_batch

        # Stack all varieties
        print('Loading data.')
        array_list = []
        for variety in self.varieties:
            df_path = self.data_file_location + variety + '_minutes_clean.csv'
            if os.path.exists(df_path):
                df = pd.read_csv(df_path, usecols=self.features)
                array_list.append(np.array(df))
        self.raw_dataset = np.stack(array_list, axis=1)

        # raw_dataset shapes like (n_dataset, n_varieties, n_features)
        self.n_dataset = self.raw_dataset.shape[0]
        self.n_varieties = self.raw_dataset.shape[1]
        self.n_features = self.raw_dataset.shape[2]
        print('%d datalines loaded with %d varieties and %d features.' % (self.n_dataset, self.n_varieties, self.n_features))

        # Split dataset into train set and test set
        self.n_train = int(self.train_data_rate * self.n_dataset)
        self.n_test = self.n_dataset - self.n_train

        self.train_dataset = self.raw_dataset[:self.n_train, :, :]
        self.test_dataset = self.raw_dataset[self.n_train:, :, :]
        print('Train and test dataset split, n_train: %d, n_test: %d.' % (self.n_train, self.n_test))

        # Set initial matrix w for train and test
        self.train_matrix_w = np.ones((self.n_train, self.n_varieties)) / self.n_varieties
        self.test_matrix_w = np.ones((self.n_test, self.n_varieties)) / self.n_varieties

    def next_batch(self):
        #Stack dataset on the 1st direction of moving time window
        rand_i = random.randint(1, self.n_train - self.n_timesteps - self.n_batch)
        array_list = []
        for i in range(self.n_batch):
            array_list.append(self.train_dataset[rand_i + i - 1:rand_i + self.n_timesteps + i,:,:])
        input_data = np.stack(array_list, axis=0) # [n_batch, n_timesteps + 1, n_varieties, n_features]

        input_x = input_data[:,:-1,:,:] / input_data[:,-2,None,:,0,None] # [n_batch, n_timesteps, n_varieties, n_features]
        input_y = input_data[:,-1,:,0] / input_data[:,-2,:,0] # [n_batch, n_varieties]
        # Get last_w for training
        last_w = self.train_matrix_w[rand_i + self.n_timesteps - 1:rand_i + self.n_timesteps + self.n_batch - 1,:]
        return rand_i, input_x, input_y, last_w

    def set_w(self, rand_i, output_w):
        # Put w into train_matrix_w
        self.train_matrix_w[rand_i + self.n_timesteps:rand_i + self.n_timesteps + self.n_batch,:] = output_w
