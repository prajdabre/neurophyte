#!/usr/bin/env python
"""nmt_train.py: The main program that runs the NMT training pipeline by calling appropriate modules. META: The name of this module draws inspiration from a fallen warrior named mt_train. mt_train had humble origins and due to excellent mentors it grew up to be an excellent warrior. It served millions of people and endured 10 years of rigourous training. Alas, what the Joker said was true that 'You think that they love you? No, they endure you because they need you. The day they don't they will cast you out like a leper.' A grand Persian who was always green with envy because of how deep learning worked took in a new protege and started raising it. This new protege turned out to be a genius and within a few months it showed that mt_train would never be able to compete no matter the amount of training. The Persian loved this new protege of his and issued mt_train to be abandoned and thus mt_train was shunned and now rots away in the pits of despair. Its once magnificent powers now wane as its various parts are in a constant state of atrophy. The only time mt_train sees the light is when some unsuspecting wayfarer, who speaks in a foreign tongue that the Persians new protege has not yet learned, seeks succour. Fear not mt_train, your death shall be remembered for your name shall be immortalised just in the same way Jesus was resurrected on the third day but with greater powers. Watch as I bring you back to life albeit in a new body."""

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


import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("models:nmt:nmt_train")
log.setLevel(logging.INFO)





if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Run the NMT training pipeline.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--data_extension", help="The location of the json objects which contain all relevant information about the training data .")
	parser.add_argument(
		"--save_extension", help="The training source corpora files.")
	parser.add_argument(
		"--learning_rate", default=0.5,  help="The training target corpora files.")
	parser.add_argument(
		"--max_gradient_norm", default=5.0,  help="Clip gradients to this norm.")
	parser.add_argument(
		"--batch_size", default=64,  help="The batch size used for training.")
	parser.add_argument(
		"--num_of_rnn_layers", default=3,  help="The number of RNN layers to stack for encoder and decoder.")
	parser.add_argument(
		"--rnn_type", default="lstm", choices = ["lstm", "gru", "basic"], help="The type of RNN to use.")
	parser.add_argument(
		"--residual_connections", default=False, action="store_true", help="Should we have residual connections between RNN layers?.")
	parser.add_argument(
		"--evaluate_every", default=200,  help="The number of iterations after which to evaluate the model on the dev set to save checkpoints.")
	parser.add_argument(
		"--anneal_after", default=2, help="The number of epochs after which to start annealing. In the current model annealing implies halving the learning rate after a prespecified number of iterations.")
	parser.add_argument(
		"--num_iterations_to_halve_lr", default=100000, help="The number of iterations after which the learning rate should be halved.")
	parser.add_argument(
		"--optimization_algorithm", default="adam", choices=["adam", "sgd", "rmsprop", "adagrad"], help="The optimization algorithm to use. Currently the options are: adam (default), sgd (stochastic gradient descent), adagrad and rmsprop.")
	parser.add_argument(
		"--embedding_size", default=1024, help="The dimensionality of the embeddings.")
	parser.add_argument(
		"--rnn_state_size", default=1024, help="The size of the hidden state for the RNN.")
	parser.add_argument(
		"--attention_hidden_layer_size", default=512, help="The size of the hidden state for the attention mechanism.")
	args = parser.parse_args()
	
	log.info("Reading data.")

	training_data_path = args.data_extension + ".data"
	vocabulary_data_path = args.data_extension + ".voc"
	training_data_config_path = args.data_extension + ".config"

	training_data = json.load(gzip.open(training_data_path, "rb"))
	vocabulary_data = json.load(open(vocabulary_data_path))
	training_data_config = json.load(open(training_data_config_path))

	train, dev, test = training_data
	src_vocs, tgt_vocs = vocabulary_data

	log.info("Data loaded.")
	log.info("Training data configuration is as follows: %s" % str(training_data_config))
	log.info("Now setting up the NMT training pipeline.")

	task_type = training_data_config.task_type

	if task_type == "basic":
		log.info("Learning a basic encoder decoder model with attention.") # Note: A basic model is essentially a MLNMT model with a single source and target.
	elif task_type == "multisource":
		log.info("Learning a multisource encoder decoder model with attention.") # TODO (Raj): Make sure that the multisource model takes parameters which allow it to have shared encoders and/or attentions (with or without attention control) and/or vocabularies. 
	elif task_type == "multilingual_multiway":
		log.info("Learning a multilingual multiway encoder decoder model with attention.") # TODO (Raj): Make sure that the multilingual multiway model takes parameters which allow it to have shared encoders and/or attentions and/or vocabularies.
		# Note: For now, I wont implement the same exact model that Orhan et al. (2016) did cause to me its overkill. What I will do is N encoders and M decoders with a shared attention which is used as it is without any special treatment.