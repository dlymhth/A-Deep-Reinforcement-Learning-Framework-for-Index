# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:21:31 2019

@author: dlymhth
"""

class Parameters:

    def __init__(self):
        # File location
        self.data_file_location = 'C:/HTH/DFIT/reinforcement/data/'
        self.model_file_location = 'C:/HTH/DFIT/reinforcement/model/model_0'
        self.figure_file_location = 'C:/HTH/DFIT/reinforcement/'

        # Input data
        self.varieties = ['a', 'i', 'j', 'jm',
                          'm', 'p', 'y']
        self.features = ['new.price',
                         'last.minute.high.price',
                         'last.minute.low.price']
        self.train_data_rate = 0.7

        # Model structure
        self.n_timesteps = 50
        self.n_batch = 32

        self.height_cov1 = 3
        self.n_cov1_core = 2
        self.height_cov2 = self.n_timesteps - self.height_cov1 + 1
        self.n_cov2_core = 20
        self.height_cov3 = self.n_cov2_core + 1

        # Training
        self.n_epochs = int(10e5)
        self.display_step = int(10e2)

        self.start_learning_rate = 0.001
        self.decay_steps = 10e3
        self.decay_rate = 0.1

        self.commission_rate = 0.001

parameters = Parameters()
