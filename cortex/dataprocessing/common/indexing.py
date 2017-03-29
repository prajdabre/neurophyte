#!/usr/bin/env python
"""indexing.py: A collection of methods to read a file and compute word and line statistics and return a dictionary and an indexer to access the file contents."""
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

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("dataprocessing:common:indexing")
log.setLevel(logging.INFO)

class Indexer:
	def __init__(self, file_path):
		log.info("Initializing the indexer.")
		log.info("File location: %s" % file_path)
		self.file_path = file_path
		self.dictionary = {}
		self.word_to_line_map = defaultdict(set)
		self.indexed_file = []
		self.num_lines = 0
		self.num_words = 0
		self.avg_words_per_line = 0
		self.sentence_length_distribution = defaultdict(int)

	def convert_line_to_id_sequence(self, input_line):
		"""
			Accept a line/sentence as string input and return a list of ids representing the line/sentence.
		"""
		input_line = ["unique#BOS#euqinu"] + input_line.strip().split(" ") + ["unique#EOS#euqinu"]
		id_seq = []

		for word in input_line:
			if self.dictionary.has_key(word):
				id_seq.append(self.dictionary[word])
			else:
				id_seq.append(self.dictionary["unique#UNK#euqinu"])

		return id_seq



def generate_dictionary(indexer_obj, max_dictionary_size = 32000):
	"""generate_dictionary: Accept an indexer object of class Indexer and a maximum dictionary size populate it with the various file statistics and a dictionary.
	args:
		indexer_obj: An object of class Indexer.
	returns:
		indexer_obj: Populated with the dictionary and statistics.
	"""
	log.info("Opening file.")
	file_to_index = io.open(indexer_obj.file_path, encoding = "utf-8")
	dictionary_list = ["unique#BOS#euqinu", "unique#EOS#euqinu", "unique#UNK#euqinu"]
	dictionary_counter = Counter()
	indexer_obj.dictionary["unique#BOS#euqinu"] = 0
	indexer_obj.dictionary["unique#EOS#euqinu"] = 1
	indexer_obj.dictionary["unique#UNK#euqinu"] = 2
	max_len = 0
	log.info("Collecting count statistics.")
	for line in file_to_index:
		indexer_obj.num_lines += 1
		if indexer_obj.num_lines % 10000 == 0:
			log.info("Read %s lines so far" % indexer_obj.num_lines)
		line = line.strip().split(" ")
		line_len = len(line)
		indexer_obj.num_words += line_len
		
		residue = 10 - line_len % 10
		if line_len + residue > max_len:
			max_len = line_len + residue

		indexer_obj.sentence_length_distribution[line_len + residue] += 1

		for word in line:
			dictionary_counter[word] += 1
	
	log.info("Read %d lines in total." % indexer_obj.num_lines)

	indexer_obj.avg_words_per_line = 1.0 * indexer_obj.num_words / indexer_obj.num_lines
	
	log.info("Average number of lines per line: %f." % indexer_obj.avg_words_per_line)

	log.info("Sentence length statistics:")
	for len_ind in xrange(10, max_len + 10, 10):
		log.info("%d sentences are in the length range of %d to %d constituting %f percent of the corpus." % (indexer_obj.sentence_length_distribution[len_ind], len_ind - 10, len_ind, indexer_obj.sentence_length_distribution[len_ind] * 100.0 / indexer_obj.num_lines))
	log.info("Taking top %d most frequent words from dictionary." % max_dictionary_size)
	dictionary_counter = dictionary_counter.most_common(max_dictionary_size)

	for word, _ in dictionary_counter:
		indexer_obj.dictionary[word] = len(dictionary_list)
		dictionary_list.append(word)
	log.info("Dictionary is ready. Size is %d tokens. The first 3 are reserved for beginning, end and unknown word tokens." % len(dictionary_list))
	log.info("Closing file.")
	file_to_index.close()
	return indexer_obj

def generate_index(indexer_obj):
	"""generate_index: Accept an indexer object of class Indexer and index the file for which it is created.
	args:
		indexer_obj: An object of class Indexer.
	returns:
		indexer_obj: Populated with a map for the id of the word to the line containing it and the sentences in the file with words replaced with the ids.
	"""
	log.info("Opening file.")
	file_to_index = io.open(indexer_obj.file_path, encoding = "utf-8")
	log.info("Indexing.")
	line_counter = 0
	word_counter = 0

	for line in file_to_index:
		line_counter += 1
		if line_counter % 10000 == 0:
			log.info("Read %s lines so far" % line_counter)
		id_seq = indexer_obj.convert_line_to_id_sequence(line)
		indexer_obj.indexed_file.append(id_seq)
		word_counter += len(id_seq[1:-1])

		for id_in_seq in id_seq[1:-1]:
			indexer_obj.word_to_line_map[id_in_seq].add(line_counter - 1)

	assert line_counter == indexer_obj.num_lines
	assert word_counter == indexer_obj.num_words

	log.info("Indexing completed.")
	
	log.info("Closing file.")
	file_to_index.close()
	return indexer_obj