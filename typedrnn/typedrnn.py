"""This module provides implementations of strongly typed RNN
architectures as per Balduzzi and Ghifary. The goal is to implement them in a
manner consitent with Tensorflow's RNN helpers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class StronglyTypedRNNCell(tf.nn.rnn_cell.RNNCell):
    """Strongly Typed vanilla RNN."""

    def __init__(self, num_units, input_size=None,
                 nonlinearity=tf.nn.sigmoid,
                 initializer=tf.random_normal_initializer(0.0, 0.1)):
        """
        Construct an RNN cell factory.

        Args:
            num_units (int): the number of hidden units.
            input_size (Optional[int]): the number of inputs, defaults to
                num_units.
            nonlinearity (callable): the nonlinearity to squash f. Defaults to
                tf.nn.sigmoid to keep it in [0,1].
            initializer (Optional): initializer to use for the weight matrices.
        """
        self._input_size = num_units if not input_size else input_size
        self._num_units = num_units
        self._nonlin = nonlinearity
        self._initializer = initializer

    @property
    def nonlinearity(self):
        return self._nonlin

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        r"""Run the RNN cell on inputs starting from the given state.

        Implements the Strongly Typed vanilla RNN given in Balduzzi and
        Ghifary (2016):

        ..math::

            \mathbf{z}_t = \mathbf{W}\mathbf{x}_t
            \mathbf{f}_t = \sigma\left(\mathbf{V}\mathbf{x}_t + \mathbf{b}
            \right)
            \mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} +
            \left(1 - \mathbf{f}_t\right) \odot \mathbf{z}_t.

        Where :math:`\mathbf{h}_t` is both the output and the hidden state.

        Args:
            inputs: 2D tensor with shape [batch_size X self.input_size].
            state: 2D Tensor with shape [batch_size x self.state_size].
            scope: VariableScope for the created subgraph, defaults to class
                name.

        Returns:
            A pair containing:
            - Output: a 2D Tensor with shape [batch_size x self.output_size]
            - New state: a 2D Tensor with shape [batch_size x self.state_size].
        """
        with tf.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):
            W = tf.get_variable('W', [self._input_size, self._num_units])
            V = tf.get_variable('V', [self._input_size, self._num_units])
            b = tf.get_variable('b', [self._num_units])
            z = tf.matmul(inputs, W)
            f = self._nonlin(tf.matmul(inputs, V) + b)
            output = f * state + (1-f) * z
        return output, output
