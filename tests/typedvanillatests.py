"""Tests for the Strongly Typed Vanilla RNN implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import unittest

import tensorflow as tf
import numpy as np

from typedrnn.typedrnn import StronglyTypedRNNCell

class TypedRNNTests(unittest.TestCase):
    """some tests, at first to make sure all the dims line up and it seems
    to be going OK."""

    def setUp(self):
        """start a session to use for testing"""
        self.sess = tf.Session()

    def test_runs_on_some_data(self):
        """Just make sure it doesn't throw anything horrid"""
        # make some fake data, 20 batches of length 30 and 40 features per step
        inputs = [
            tf.Variable(
                np.random.randn(20,40).astype(
                    np.float32)) for _ in xrange(30)]
        cell = StronglyTypedRNNCell(50, 40)
        initial_state = tf.ones([cell.state_size])
        net, state = tf.nn.rnn(
            cell,
            inputs,
            initial_state=initial_state)
        labels = [tf.Variable([0]) for _ in xrange(30)]
        
        return False
