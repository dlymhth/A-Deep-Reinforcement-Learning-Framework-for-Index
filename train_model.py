# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:05:39 2019

@author: dlymhth
"""

import tensorflow as tf

class NNAgent:

    def __init__(self, parameters):

        self.n_batch = parameters.n_batch
        self.n_timesteps = parameters.n_timesteps
        self.n_varieties = len(parameters.varieties)
        self.n_features = len(parameters.features)

        self.height_cov1 = parameters.height_cov1
        self.n_cov1_core = parameters.n_cov1_core
        self.height_cov2 = parameters.height_cov2
        self.n_cov2_core = parameters.n_cov2_core
        self.height_cov3 = parameters.height_cov3

        self.n_epochs = parameters.n_epochs
        self.display_step = parameters.display_step

        self.start_learning_rate = parameters.start_learning_rate
        self.decay_steps = parameters.decay_steps
        self.decay_rate = parameters.decay_rate

        self.commission_rate = parameters.commission_rate

        # tf Graph
        # input_x shape [32, 50, 7, 3]
        self.X = tf.placeholder('float32', [self.n_batch, self.n_timesteps,
                                            self.n_varieties, self.n_features])

        # Cov 1 core: [height: 3, width: 1, in_channels: 3, out_channels: 2]
        cov1_core = tf.Variable(tf.random_normal([self.height_cov1, 1,
                                                  self.n_features, self.n_cov1_core]))
        raw_cov_layer1 = tf.nn.conv2d(input=self.X, filter=cov1_core,
                                      strides=[1, 1, 1, 1], padding='VALID') # [32, 48, 7, 2]
        cov_layer1 = tf.nn.relu(raw_cov_layer1)

        # Cov 2 core: [height: 48, width: 1, in_channels: 2, out_channels: 20]
        cov2_core = tf.Variable(tf.random_normal([self.height_cov2, 1,
                                                  self.n_cov1_core, self.n_cov2_core]))
        raw_cov_layer2 = tf.nn.conv2d(input=cov_layer1, filter=cov2_core,
                                      strides=[1, 1, 1, 1], padding='VALID') # [32, 1, 7, 20]
        cov_layer2 = tf.nn.relu(raw_cov_layer2)

        self.last_w = tf.placeholder('float32', [self.n_batch, self.n_varieties]) # [32, 7]
        last_w_1 = tf.expand_dims(self.last_w, axis=1) # [32, 1, 7]
        last_w_2 = tf.expand_dims(last_w_1, axis=3) # [32, 1, 7, 1]
        concat_layer = tf.concat([cov_layer2, last_w_2], axis=3) # [32, 1, 7, 21]

        # Cov 3 core: [height: 1, width: 1, in_channels: 21, out_channels: 1]
        cov3_core = tf.Variable(tf.random_normal([1, 1, self.height_cov3, 1]))
        raw_cov_layer3 = tf.nn.conv2d(input=concat_layer, filter=cov3_core,
                                      strides=[1, 1, 1, 1], padding='VALID') # [32, 1, 7, 1]
        self.output_w = tf.nn.softmax(tf.squeeze(raw_cov_layer3), axis=1) # [32, 7]

        # Define loss and optimizer
        # input_y shape [32, 7]
        self.y = tf.placeholder('float32', [self.n_batch, self.n_varieties])
        omega_y = tf.reduce_sum(tf.multiply(tf.squeeze(self.y), self.output_w), axis=1)
        
        self.loss = -tf.reduce_mean(tf.log(omega_y))

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=self.start_learning_rate,
                                                        global_step=self.global_step,
                                                        decay_steps=self.decay_steps,
                                                        decay_rate=self.decay_rate)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.config = tf.ConfigProto(log_device_placement=False,
                                     allow_soft_placement=True)

    def train(self, dataset):

        init = tf.global_variables_initializer()
        sess = tf.Session(config=self.config)

        # Run the initializer
        sess.run(init)

        # Training
        for epoch in range(self.n_epochs):
            rand_i, input_x, input_y, last_w = dataset.next_batch()
            _, loss, output_w = sess.run([self.train_op, self.loss, self.output_w],
                                         feed_dict={self.X: input_x,
                                                    self.y: input_y,
                                                    self.last_w: last_w})
            # Write output_w into train_matrix_w
            dataset.set_w(rand_i, output_w)
            if epoch % self.display_step == 0:
                print(loss)
                print(output_w[0])

        sess.close()
        print('Training done.')