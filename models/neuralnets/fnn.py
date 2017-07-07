from __future__ import print_function
from ..abstract_model import AbstractModel
from ..cost import mse
import numpy as np
import tensorflow as tf


class FNN(AbstractModel):
    """

    """

    def __init__(self, n_inputs=26, n_outputs=1):
        """

        Args:
            n_inputs:
            n_outputs:
        """

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.x = tf.placeholder("float", [None, self.n_inputs])
        self.y = tf.placeholder("float", [None, self.n_outputs])
        self.batch_size = tf.placeholder("float", None)

        self.sizes = [n_inputs, 13, 13, n_outputs]
        self.weights = []
        self.biases = []
        self.layers = []

        for i in xrange(len(self.sizes) - 1):
            # Weight initialization based on Torch7 Linear.lua and CS231n
            # https://github.com/torch/nn/blob/master/Linear.lua
            # https://cs231n.github.io/neural-networks-2/#init
            self.weights.append(tf.Variable(tf.random_normal([self.sizes[i], self.sizes[i + 1]],
                                                             stddev=1/np.sqrt(self.sizes[i]))))
            self.biases.append(tf.Variable(tf.random_normal([self.sizes[i + 1]],
                                                            stddev=np.sqrt(2.0/self.sizes[i+1]))))
            if len(self.layers) == 0:   # First layer
                self.layers.append(tf.add(tf.matmul(self.x, self.weights[i]), self.biases[i]))
                self.layers.append(tf.nn.relu(self.layers[len(self.layers) - 1]))
                # self.layers.append(tf.nn.dropout(self.layers[len(self.layers) - 1], 0.5))
            elif i == len(self.sizes) - 2:  # Last layer
                self.layers.append(
                    tf.add(tf.matmul(self.layers[len(self.layers) - 1], self.weights[i]), self.biases[i]))
                self.layers.append(tf.nn.relu(self.layers[len(self.layers) - 1]))
                # self.layers.append(tf.nn.dropout(self.layers[len(self.layers) - 1], 0.5))
            else:
                self.layers.append(
                    tf.add(tf.matmul(self.layers[len(self.layers) - 1], self.weights[i]), self.biases[i]))
                self.layers.append(tf.nn.relu(self.layers[len(self.layers) - 1]))
                # self.layers.append(tf.nn.dropout(self.layers[len(self.layers) - 1], 0.5))
        self.model = self.layers[len(self.layers) - 1]

        # Define Cost Function
        # self.cost = tf.divide(tf.reduce_mean(tf.abs(tf.sub(self.y, self.model))), self.batch_size)
        self.cost = mse(self.y, self.model, self.batch_size)

    def train(self, X, y, learning_rate=0.5, epochs=20, batch_size=10, display_step=1):
        """

        Args:
            X:
            y:
            learning_rate:
            epochs:
            batch_size:
            display_step:

        Returns:

        """

        # Define optimizer
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Initialize saver
        saver = tf.train.Saver()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            for epoch in xrange(epochs):
                avg_cost = 0.
                total_batch = int(len(X) / batch_size)

                idx = np.random.permutation(X.shape[0])

                # Loop over all batches
                for i in xrange(total_batch):
                    batch_x = X[idx[i * batch_size:i * batch_size + batch_size]]
                    batch_y = y[idx[i * batch_size:i * batch_size + batch_size]]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, self.cost], feed_dict={self.x: batch_x,
                                                                       self.y: batch_y,
                                                                       self.batch_size: batch_size})

                    # Compute average loss
                    avg_cost += c
                # Display logs per epoch step
                if (epoch + 1) % display_step == 0 or epoch == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "avg_cost=",
                          "{:.5f}".format(avg_cost/total_batch))
            saver.save(sess, "/tmp/model.ckpt")
            print("Optimization Finished!")

    def accuracy(self, X, y, file="/tmp/model.ckpt"):
        """

        Args:
            X:
            y:

        Returns:

        """

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        avg_cost = 0.
        cost = []

        with tf.Session() as sess:
            saver.restore(sess, file)
            total_batch = 0
            for i in xrange(len(X)):
                total_batch += 1
                xdata = np.reshape(X[i], (-1, X[i].shape[0]))
                ydata = np.reshape(y[i], (-1, y[i].shape[0]))
                cost.append(sess.run(self.cost, feed_dict={self.x: xdata, self.y: ydata,
                                                           self.batch_size: X[i].shape[0]}))

                avg_cost += cost[len(cost) - 1]

        # Compute average loss
        avg_cost = avg_cost / total_batch
        return cost, avg_cost

    def predict(self, X, file="/tmp/model.ckpt"):
        """

        Args:
            X:

        Returns:

        """

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        output = []

        with tf.Session() as sess:
            saver.restore(sess, file)
            for i in xrange(len(X)):
                x = np.reshape(X[i], (-1, X[i].shape[0]))
                output.append(sess.run(self.model, feed_dict={self.x: x})[0][0])
                # Compute average loss

        return output
