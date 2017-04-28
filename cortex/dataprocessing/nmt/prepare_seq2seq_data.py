#!/usr/bin/env python
"""prepare_seq2seq_data.py: A collection of classes and methods to convert a collection of parallel corpora into a format which will be fed to the NMT training and testing pipeline."""
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
from cortex.dataprocessing.common.indexing import *

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("dataprocessing:common:prepare_seq2seq_data")
log.setLevel(logging.INFO)

class Seq2SeqData:
	def __init__(self, args):
		log.info("Sequence to Sequence Data Creation begins.")
		self.pairwise_training_data = defaultdict()
		self.pairwise_dev_data = defaultdict()
		self.pairwise_test_data = defaultdict()
		self.source_dictionaries = defaultdict()
		self.target_dictionaries = defaultdict()
		self.meta = args

	def save_self(self):
		log.info("Saving dictionaries to file.")
		json.dump([self.source_dictionaries, self.target_dictionaries], open(self.meta.save_path + ".voc", "w"), indent=2, separators=(',', ': '))
		log.info("Dictionaries saved.")
		log.info("Saving training, dev and test data to file.")
		json.dump([self.pairwise_training_data, self.pairwise_dev_data, self.pairwise_test_data], gzip.open(self.meta.save_path + ".data", "wb"), indent=2, separators=(',', ': '))
		log.info("Training, dev and test data saved.")
		log.info("Saving configuration.")
		json.dump(args.__dict__, open(self.meta.save_path + ".config", "w"), indent=2, separators=(',', ': '))
		log.info("Configuration saved.")

def populate_seq2seq_data(seq2seqdata, language_pairs, indexed_sources, indexed_targets, type_data = "train"):
	if type_data == "train":
		for source in indexed_sources:
			seq2seqdata.source_dictionaries[source] = indexed_sources[source].dictionary
		for target in indexed_targets:
			seq2seqdata.target_dictionaries[target] = indexed_targets[target].dictionary

		for source in indexed_sources:
			indexed_sources[source].indexed_files.reverse()
		for target in indexed_targets:
			indexed_targets[target].indexed_files.reverse()

		for language_pair in language_pairs:
			source = language_pair.split("-")[0]
			target = language_pair.split("-")[1]
			seq2seqdata.pairwise_training_data[language_pair] = [[x,y] for x,y in zip(indexed_sources[source].indexed_files.pop(), indexed_targets[target].indexed_files.pop())]
	else:
		for source in indexed_sources:
			indexed_sources[source].indexed_files.reverse()
		for target in indexed_targets:
			indexed_targets[target].indexed_files.reverse()

		for language_pair in language_pairs:
			source = language_pair.split("-")[0]
			target = language_pair.split("-")[1]
			data = [[x,y] for x,y in zip(indexed_sources[source].indexed_files.pop(), indexed_targets[target].indexed_files.pop())]
			if type_data == "dev":
				seq2seqdata.pairwise_dev_data[language_pair] = data
			else:
				seq2seqdata.pairwise_test_data[language_pair] = data
	return seq2seqdata
	

def populate_indexers(source_languages, target_languages, src_corpora, tgt_corpora, max_src_vocab_size, max_tgt_vocab_size, src_dictionaries = None, tgt_dictionaries = None):
	sources = defaultdict(list)
	targets = defaultdict(list)

	for i, source in enumerate(source_languages):
		sources[source].append(src_corpora[i])
	for i, target in enumerate(target_languages):
		targets[target].append(tgt_corpora[i])

	indexed_sources = defaultdict()
	indexed_targets = defaultdict()

	for source in sources:
		indexed_sources[source] = MultiIndexer(sources[source])
		indexed_sources[source] = generate_dictionary(indexed_sources[source], max_src_vocab_size)
		if src_dictionaries is not None:
			indexed_sources[source].dictionary = src_dictionaries[source]
		indexed_sources[source] = generate_index(indexed_sources[source])

	for target in targets:
		indexed_targets[target] = MultiIndexer(targets[target])
		indexed_targets[target] = generate_dictionary(indexed_targets[target], max_tgt_vocab_size)
		if tgt_dictionaries is not None:
			indexed_targets[target].dictionary = tgt_dictionaries[target]
		indexed_targets[target] = generate_index(indexed_targets[target])
	return indexed_sources, indexed_targets

def get_source_languages(lang_pairs):
	return [pair.split("-")[0] for pair in lang_pairs]

def get_target_languages(lang_pairs):
	return [pair.split("-")[1] for pair in lang_pairs]

def generate_multisource_data(args):
	log.info("Generating data.")
	train_source_languages = get_source_languages(args.train_language_pairs)
	dev_source_languages = get_source_languages(args.dev_language_pairs)
	test_source_languages = get_source_languages(args.test_language_pairs)
	train_target_languages = get_target_languages(args.train_language_pairs)
	dev_target_languages = get_target_languages(args.dev_language_pairs)
	test_target_languages = get_target_languages(args.test_language_pairs)

	assert len(train_source_languages) == len(dev_source_languages) == len(test_source_languages)
	assert len(set(train_target_languages)) == len(set(dev_target_languages)) == len(set(test_target_languages)) == 1
		

def generate_multilingual_data(args, type_data = "basic"):
	log.info("Generating data.")

	seq2seqdata = Seq2SeqData(args)

	train_source_languages = get_source_languages(args.train_language_pairs)
	dev_source_languages = get_source_languages(args.dev_language_pairs)
	test_source_languages = get_source_languages(args.test_language_pairs)
	train_target_languages = get_target_languages(args.train_language_pairs)
	dev_target_languages = get_target_languages(args.dev_language_pairs)
	test_target_languages = get_target_languages(args.test_language_pairs)

	if type_data == "basic":
		assert len(train_source_languages) == len(dev_source_languages) == len(test_source_languages) == 1
		assert len(set(train_target_languages)) == len(set(dev_target_languages)) == len(set(test_target_languages)) == 1
		
	log.info("Processing training data.")

	indexed_src_data, indexed_tgt_data = populate_indexers(train_source_languages, train_target_languages, args.train_src_corpora, args.train_tgt_corpora, args.max_src_vocab_size, args.max_tgt_vocab_size)
	seq2seqdata = populate_seq2seq_data(seq2seqdata, args.train_language_pairs, indexed_src_data, indexed_tgt_data)

	log.info("Training data processed.")	

	log.info("Processing dev data.")

	indexed_src_data, indexed_tgt_data = populate_indexers(dev_source_languages, dev_target_languages, args.dev_src_corpora, args.dev_tgt_corpora, args.max_src_vocab_size, args.max_tgt_vocab_size,src_dictionaries = seq2seqdata.source_dictionaries, tgt_dictionaries = seq2seqdata.target_dictionaries)
	
	seq2seqdata = populate_seq2seq_data(seq2seqdata, args.dev_language_pairs, indexed_src_data, indexed_tgt_data, type_data = "dev")

	log.info("Dev data processed.")

	log.info("Processing test data.")

	indexed_src_data, indexed_tgt_data = populate_indexers(test_source_languages, test_target_languages, args.test_src_corpora, args.test_tgt_corpora, args.max_src_vocab_size, args.max_tgt_vocab_size,src_dictionaries = seq2seqdata.source_dictionaries, tgt_dictionaries = seq2seqdata.target_dictionaries)
	
	seq2seqdata = populate_seq2seq_data(seq2seqdata, args.dev_language_pairs, indexed_src_data, indexed_tgt_data, type_data = "test")

	log.info("Test data processed.")

	seq2seqdata.save_self()


if __name__ == '__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser(description="Read in a collection of parallel corpora and convert them into a format which can be fed to an NMT pipeline.",
									 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--save_path", help="The location where all the generated data will be saved.")
	parser.add_argument(
		"--train_src_corpora", nargs = "+", help="The training source corpora files.")
	parser.add_argument(
		"--train_tgt_corpora", nargs = "+", help="The training target corpora files.")
	parser.add_argument(
		"--test_src_corpora", nargs = "+", help="The testing source corpora files.")
	parser.add_argument(
		"--test_tgt_corpora", nargs = "+", help="The testing target corpora files.")
	parser.add_argument(
		"--dev_src_corpora", nargs = "+", help="The development source corpora files.")
	parser.add_argument(
		"--dev_tgt_corpora", nargs = "+", help="The development target corpora files.")
	parser.add_argument(
		"--task_type", default = "basic", choices = ["basic", "multisource", "multilingual_multiway"], help="We can train 3 types of NMT models: basic (this can be of 3 types: 1 source 1 target, concatenated multisource to 1 target, zero shot for N sources to M targets), multisource (N sources 1 target with a single encoder for all languages where the source sentence is simply a concatenation of all the source sentences for each target sentence) and multilingual_multiway (N sources M targets with separate encoders and decoders but shared attention mechanism).")
	parser.add_argument(
		"--max_src_vocab_size", default = 32000, type = int, help="The maximum source vocabulary size. This is a number that specifies the maximum vocabulary size per encoder. This number will also be used to specify the maximum number of word pieces generated by the segmentation mechanism which is assumed to be BPE. Thus in effect the actual vocabulary size per encoder will be smaller than this limit.")
	parser.add_argument(
		"--max_tgt_vocab_size", default = 32000, type = int, help="The maximum target vocabulary size. This is a number that specifies the maximum vocabulary size per decoder. This number will also be used to specify the maximum number of word pieces generated by the segmentation mechanism which is assumed to be BPE. Thus in effect the actual vocabulary size per decoder will be smaller than this limit.")
	parser.add_argument(
		"--train_language_pairs", nargs = "+", help="The training language pairs for the NMT model. Example: fr-en fr-de de-fr")
	parser.add_argument(
		"--dev_language_pairs", nargs = "+", help="The development language pairs for the NMT model. Example: fr-en fr-de de-fr")
	parser.add_argument(
		"--test_language_pairs", nargs = "+", help="The testing language pairs for the NMT model. Example: fr-en fr-de de-fr")
	args = parser.parse_args()
	
	if args.task_type == "multisource":
		log.info("Preparing data for multisource NMT model.")
		all_data = generate_multisource_data(args)

	if args.task_type == "basic":
		log.info("Preparing data for single source single target NMT model.")
		all_data = generate_multilingual_data(args, type_data = "basic")	

	if args.task_type == "multilingual_multiway":
		log.info("Preparing data for multilingual multiway NMT model.")
		all_data = generate_multilingual_data(args, type_data = "multilingual_multiway")	