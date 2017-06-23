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


def translate_no_jutsu():
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Train an 'attention is all you need' NMT model.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--save_path", help="The location where all the generated data will be saved.")
	parser.add_argument(
		"--training_data", help="The prefix of the files containing all the dictionaries, training data along with the dev and test data and the data creation config.")
	parser.add_argument(
		"--layer_size", default = 512, help="This will single handedly represent the dimensionality of embedding as well as the number of nodes for the feed forward layers for 'attention is all you need'. In case you plan on running a RNN Encoder-Decoder model then dont bother with this parameter. At a later point I will make this more flexible so that I can have embeddings and feed forward layers with different dimensionality. All I need to do is have linear projection layers between them.")
	parser.add_argument(
		"--num_layers", default = 1, help="The number of LSTM or FF layer stacks you want to use.")
	parser.add_argument(
		"--use_multi_head_attention", default = False, action = store_true, help="Should we use multi-head attention?.")
	parser.add_argument(
		"--evaluate_every", default = 200, help="By default, every 200 iterations the model will be evaluated on the dev and test data (optional) and the BLEU scores and losses will be computed. This will also print 8 sample translations from the current training batch.")
	parser.add_argument(
		"--optimizers", default = "adam", choices = ["adam", "rmsprop", "sgd", "momentum"], help="The parameter optimization strategy you want to use. Adam is default.")
	parser.add_argument(
		"--regularization", default = 0.000001, help="The amount of weight decay or regularization you want to apply.")
	parser.add_argument(
		"--initial_adam_lr", default = 0.001, help="The initial learning rate for adam.")
	parser.add_argument(
		"--initial_lr", default = 0.001, help="The initial learning rate for sgd or rmsprop or momentum optimizers.")
	parser.add_argument(
		"--momentum", default = 0.01, help="The momentum for rmsprop or momentum optimizers.")
	parser.add_argument(
		"--num_epochs", default = 5, help="The number of epochs for which training should be run.")
	parser.add_argument(
		"--sgd_switch", default = 40000, help="The number of iterations after which we should switch to SGD.")
	parser.add_argument(
		"--sgd_annealing", default = 100000, help="The number of iterations after which we should halve the learning rate.")
	parser.add_argument(
		"--halve_lr_every", default = 20000, help="The number of iterations after which we should start halving the learning rate.")
	parser.add_argument(
		"--batching_strategy", default = "basic", choices = ["basic", "bucketed"], help="The batching logic to be used. The basic method is to not group sentences by approximate length and seems to work well but the bucketing method is touted to perform much better.")
	parser.add_argument(
		"--checkpoint_every", default = 4000, help="Every 4000 (default) iterations a checkpoint of the model will be saved.")
	parser.add_argument(
		"--load_model", help="The path to the model to be loaded before training. This is for the following reasons: 1. Resuming training. 2. Transfer learning. 3. To see how NMT behaves weirdly when you do a ramdom restart of the training by keeping the model parameters but resetting the optimizer.")
	parser.add_argument(
		"--load_optimizer_state", help="The path to the saved optimizer state to be loaded. Dont use this if you want to do 2 and 3 mentioned in the load_model argument.")



	args = parser.parse_args()

if __name__ == '__main__':
    translate_no_jutsu()