# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 09:53:23 2019

@author: dlymhth
"""

import read_data
import train_model

from set_parameters import parameters

if __name__ == '__main__':

    dataset = read_data.Dataset(parameters)

    model = train_model.NNAgent(parameters)
    model.train(dataset)
