#!/usr/bin/env python
"""data_iterators.py: A class and relevant methods to iterate over a given dataset."""
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

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("utilities:nmt:data_iterators")
log.setLevel(logging.INFO)

class DataHandler:
	def __init__(self, data):
		log.info("Initializing the data handler.")
		self.data = data
		self.buckets = [[5,10], [10, 15], [20, 25], [40, 50], [50, 60], [60, 70], [70, 80], [80, 90], [90, 100]]
		self.bucketed_data = {}
		for src_len, _ in self.buckets:
			self.bucketed_data[src_len] = []
		self.epochs_completed = 0

	def random_shuffle(self):
		log.info("Randomly shuffling the input data.")
		self.data = random.shuffle(self.data)
		log.info("Random shuffling complete.")

	def filter_long_sentences(self, max_seq_len = 100, shorten_sequences = False):
		log.info("Filtering all sequences with source or target sequences longer than %s units." % str(max_seq_len))
		new_data = []
		filter_count = 0
		for src_seq, tgt_seq in self.data:
			if len(src_seq) > max_seq_len or len(tgt_seq) > max_seq_len:
				filter_count += 1
				if shorten_sequences:
					new_data.append([src_seq[:max_seq_len], tgt_seq[:max_seq_len]])
			else:
				new_data.append([src_seq, tgt_seq])

		self.data = new_data
		log.info("Filtering complete. Removed/Shortened %s sequence pairs. Current number of sequence pairs %s." % (str(filter_count), str(len(self.data))))

	def sort(self, sort_type = "ascending", sort_by = "source"):
		log.info("Sorting the data in %s order in terms of lenth of %s sequences." % (sort_type, sort_by))
		reverse = False if sort_type == "ascending" else True
		sort_key = None
		if sort_by == "source":
			sort_key = lambda x: len(x[0])
		elif sort_by == "target":
			sort_key = lambda x: len(x[1])
		elif sort_by == "source+target":
			sort_key = lambda x: len(x[0]+x[1])
		self.data = sorted(self.data, reverse = reverse, key = sort_key)
		log.info("Data sorting complete.")

	def reverse_target_sequences(self):
		log.info("Reversing target sequences.")
		for pair in self.data:
			pair[1].reverse()
		log.info("Target sequences reversed.")

	def filter_data(self, max_src_len = 100, max_tgt_len = 100):
		log.info("Filtering pairs with source sequence longer than %s and target sequence longer than %s." % (str(max_src_len), str(max_tgt_len)))
		filtered_data = []
		removed_count = 0

		for pair in self.data:
			if len(pair[0]) <= max_src_len and len(pair[1]) <= max_tgt_len:
				filtered_data.append(pair)
			else:
				removed_count += 1
		log.info("Originally there were %s number of pairs and now there are %s (%s were removed)." % (str(len(self.data)), str(len(filtered_data)), str(removed_count)))
		self.data = filtered_data

	def generate_bucketed_data(self):
		log.info("Splitting data into buckets.")
		total_pairs_count = 0
		bucketed_pairs_count = 0
		self.pairs_per_bucket = {}
		for src_len, _ in self.buckets:
			self.pairs_per_bucket[src_len] = 0

		max_bucket_id = 0

		for pair in self.data:
			src_seq_len = len(pair[0])
			tgt_seq_len = len(pair[1])
			total_pairs_count = 1

			for src_len, tgt_len in self.buckets:
				if src_seq_len < src_len and tgt_seq_len < tgt_len:
					self.bucketed_data[src_len].append(pair)
					bucketed_pairs_count += 1
					self.pairs_per_bucket[src_len] += 1
					if max_bucket_id < src_len:
						max_bucket_id = src_len
					break
		log.info("Total pairs are %s and out of them %s are bucketed." % (str(total_pairs_count), str(bucketed_pairs_count)))
		log.info("The bucketwise statistics are:")
		for src_len, tgt_len in self.buckets:
			log.info("Source, target sequence lengths are %s and %s and the number of such pairs are %s." % (str(src_len), str(tgt_len), str(self.pairs_per_bucket[src_len])))
			if src_len == max_bucket_id:
				break

		log.info("Generating bucket distribution for sampling.")
		bucket_counts = []
		for src_len, _ in self.buckets:
			bucket_counts.append(self.pairs_per_bucket[src_len])
			if src_len == max_bucket_id:
				break
		
		self.bucket_distribution = [1.0 * sum(bucket_counts[:i+1])/bucketed_pairs_count for i in range(len(bucket_counts))]
		log.info("Done.")

	def get_random_bucket(self, force_random = None):
		log.info("Selecting random bucket.")
		random_number = random.random() if force_random is None else force_random
		return self.buckets[min([i for i in range(len(self.bucket_distribution)) if self.bucket_distribution[i] > random_number])][0]

	def batched_data_generator_for_bucketed_data(self, batch_size = 64, num_epochs = 5, sorted_batches = True, sort_by = "source"):
		log.info("Preparing to generate batches.")
		sort_key = None
		if sort_by == "source":
			sort_key = lambda x: len(x[0])
		elif sort_by == "target":
			sort_key = lambda x: len(x[1])
		elif sort_by == "source+target":
			sort_key = lambda x: len(x[0]+x[1])
		
		for epoch_id in range(num_epochs):
			log.info("Starting epoch %s." % str(epoch_id + 1))
			coverage_information_per_bucket = {}

			for distr_index in range(len(self.bucket_distribution)):
				if self.pairs_per_bucket[self.buckets[distr_index][0]] == 0:
					coverage_information_per_bucket[self.buckets[distr_index][0]] = [0, 0, True]
				else:
					coverage_information_per_bucket[self.buckets[distr_index][0]] = [0, batch_size, False]
				

			while True:
				if reduce(lambda x, y: x and y, [info[2] for info in coverage_information_per_bucket.values()]):
					log.info("All buckets covered. Ending epoch %s." % str(epoch_id + 1))
					self.epochs_completed += 1
					break

				random_bucket_id = self.get_random_bucket()
				if coverage_information_per_bucket[random_bucket_id][2]:
					continue

				print "Geeting from bucket:",random_bucket_id
				if coverage_information_per_bucket[random_bucket_id][1] > self.pairs_per_bucket[random_bucket_id]:
					coverage_information_per_bucket[random_bucket_id][1] = self.pairs_per_bucket[random_bucket_id]
				
				current_begin_index = coverage_information_per_bucket[random_bucket_id][0]
				current_end_index = coverage_information_per_bucket[random_bucket_id][1]
				
				current_batch = self.bucketed_data[random_bucket_id][current_begin_index:current_end_index]

				if sorted_batches:
					current_batch = sorted(current_batch, reverse = False, key = sort_key)
				
				yield current_batch

				if current_end_index == self.pairs_per_bucket[random_bucket_id]:
					coverage_information_per_bucket[random_bucket_id][2] = True
					log.info("Bucket for source sentence length %s has been exhausted." % str(random_bucket_id))
					continue
				coverage_information_per_bucket[random_bucket_id][0] = current_end_index
				coverage_information_per_bucket[random_bucket_id][1] += batch_size



	def batched_data_generator(self, batch_size = 64, num_epochs = 5, sorted_batches = True, sort_by = "source"):
		log.info("Preparing to generate batches.")
		sort_key = None
		if sort_by == "source":
			sort_key = lambda x: len(x[0])
		elif sort_by == "target":
			sort_key = lambda x: len(x[1])
		elif sort_by == "source+target":
			sort_key = lambda x: len(x[0]+x[1])
		maximum_index = len(self.data)

		for epoch_id in range(num_epochs):
			log.info("Starting epoch %s." % str(epoch_id + 1))
			current_begin_index = 0
			current_end_index = batch_size * batch_size
			while True:
				if current_end_index > maximum_index:
					current_end_index = maximum_index

				current_data_segment = self.data[current_begin_index:current_end_index]
				if sorted_batches:
					current_data_segment = sorted(current_data_segment, reverse = False, key = sort_key)
				current_begin_batch_index = 0
				current_end_batch_index = batch_size
				maximum_batch_index = len(current_data_segment)
				while True:
					if current_end_batch_index > maximum_batch_index:
						current_end_batch_index = maximum_batch_index

					yield current_data_segment[current_begin_batch_index:current_end_batch_index]
					
					if current_end_batch_index == maximum_batch_index:
						break

					current_begin_batch_index = current_end_batch_index
					current_end_batch_index += batch_size

				if current_end_index == maximum_index:
					log.info("Ending epoch %s." % str(epoch_id + 1))
					self.epochs_completed += 1
					break
				
				current_begin_index = current_end_index
				current_end_index += batch_size*batch_size

def pad_batch_and_generate_mask(batch, src_padding_id, tgt_padding_id):
	log.info("Padding batch and generating masks.")
	max_src_len = max([len(x[0]) for x in batch])
	max_tgt_len = max([len(x[1]) for x in batch])
	padded_batch = deepcopy(batch)
	batch_mask = []
	for example_id in range(len(batch)):
		src_seq = padded_batch[example_id][0]
		tgt_seq = padded_batch[example_id][1]
		src_mask = [True] * max_src_len
		tgt_mask = [True] * max_tgt_len
		src_len = len(src_seq)
		tgt_len = len(tgt_seq)
		for src_index in range(src_len, max_src_len):
			src_seq.append(src_padding_id)
			src_mask[src_index] = False
		for tgt_index in range(tgt_len, max_tgt_len):
			tgt_seq.append(tgt_padding_id)
			tgt_mask[tgt_index] = False
		batch_mask.append([src_mask, tgt_mask])
	return padded_batch, batch_mask

def convert_batch_into_feed(padded_batch, batch_mask):
	log.info("Generating feeds.")
	batch_size = len(padded_batch)
	src_seq_len = len(padded_batch[0][0])
	tgt_seq_len = len(padded_batch[0][1])

	src_feed, tgt_feed, src_weights, tgt_weights = [], [], [], []

	for src_word_index in range(src_seq_len):
		src_feed.append(np.array([padded_batch[batch_index][0][src_word_index] for batch_index in range(batch_size)], dtype=np.int32))
		src_weights.append(np.array([batch_mask[batch_index][0][src_word_index] for batch_index in range(batch_size)], dtype=np.float32))

	for tgt_word_index in range(tgt_seq_len):
		tgt_feed.append(np.array([padded_batch[batch_index][1][tgt_word_index] for batch_index in range(batch_size)], dtype=np.int32))
		tgt_weights.append(np.array([batch_mask[batch_index][1][tgt_word_index] for batch_index in range(batch_size)], dtype=np.float32))

	return src_feed, tgt_feed, src_weights, tgt_weights

def sort(self, data, sort_type = "ascending", sort_by = "source"):
	log.info("Sorting the data in %s order in terms of lenth of %s sequences." % (sort_type, sort_by))
	reverse = False if sort_type == "ascending" else True
	sort_key = None
	if sort_by == "source":
		sort_key = lambda x: len(x[0])
	elif sort_by == "target":
		sort_key = lambda x: len(x[1])
	elif sort_by == "source+target":
		sort_key = lambda x: len(x[0]+x[1])
	data = sorted(data, reverse = reverse, key = sort_key)
	log.info("Data sorting complete.")
	return data