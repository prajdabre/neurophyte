#!/usr/bin/env python
"""encoders.py: Classes and relevant methods for various encoders for NMT."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"

import collections
import logging
import codecs
import json
import operator
import os.path
import gzip
import io
import random
import itertools
from itertools import chain, combinations
from collections import defaultdict, Counter
from copy import deepcopy
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class Linear_Layers(ChainList):
	def __init__(self, num_layers = 2, layer_size = 512, dropout = 0.2, gpu = 0, activation = F.relu):
		self.layer_size = layer_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.gpu = gpu
		self.activation = activation

		for layer_id in xrange(self.num_layers):
			self.add_link(L.Linear(self.layer_size, self.layer_size))

	def __call__(self, self_attentioned_sequence, layer_id):
		batch_size, sequence_length, layer_size = self_attentioned_sequence.data.shape
		self_attentioned_sequence = F.reshape(self_attentioned_sequence, [batch_size * sequence_length, layer_size])
		self_attentioned_sequence = F.dropout(self.activation(self[layer_id](self_attentioned_sequence)))
		self_attentioned_sequence = F.reshape(self_attentioned_sequence, [batch_size ,sequence_length, layer_size])
		return self_attentioned_sequence

class Basic_Attention(ChainList):
	def __init__(self, layer_size):
		log.info("Using Basic Attention. No additional parameters will be created.")
		self.scaling_factor = 1.0 / layer_size ** (0.25)

	def __call__(self, query, key, value):
		return F.batch_matmul(F.softmax(F.batch_matmul(query, key, transb = True) * self.scaling_factor), value)

class FFEncoder(ChainList):
	def __init__(self, vocab_size = 32000, embedding_size = 512, layer_size = 512, num_layers = 2, dropout = 0.2, gpu = 0):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.layer_size = layer_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.gpu = gpu


class BasicRNNEncoder(Chain):
	def __init__(self, vocab_size = 32000, embedding_size = 512, layer_size = 512, num_layers = 2, dropout = 0.2, gpu = 0, rnn_type = L.NStepLSTM):
		log.info("Creating the Encoder.")

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.layer_size = layer_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.gpu = gpu
		
		super(BasicEncoder, self).__init__(
            embedding = L.EmbedID(vocab_size, embedding_size),
            stackedrnn = StackedRNN(num_layers, embedding_size, lstm_size, dropout)
        )

        initial_cell_state = None
        initial_hidden_state = None

	def __call__(self, input_sequences):
		#embeded_sequences = self.embedding(F.concat(F.reshape(input_sequences, [-1, 1]), axis = 1))
		embeded_sequences = []

		for individual_input in input_sequences:
			embeded_sequences.append(self.embedding(individual_input))

		stacked_h, stacked_c, _ = self.stackedrnn(None, None, embeded_sequences)