#!/usr/bin/env python
"""data_iterators_test.py: Tests for the methods and classes in data_iterators.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
from data_iterators import *
import io
import uuid
import logging

logging.basicConfig()
log = logging.getLogger("utilities:nmt:data_iterators_test")
log.setLevel(logging.INFO)

class TestDataIterators(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestDataIterators, self).__init__(*args, **kwargs)
		log.info("Initializing the test.")
		self.seq2seqdata = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]],[[1,2],[1]]]
		self.seq2seqdata_longer = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]],[[1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]]]

	def test_reverse_target_sequences(self):
		log.info("Testing reversing target sequences.")
		reversed_target_data_reference = [[[1,2,3],[2,1]],[[1,2,3,4,5],[7,6,5,4,3,2,1]],[[1],[2,1]],[[1,2],[1]]]
		data_iterator = DataHandler(list(self.seq2seqdata))
		data_iterator.reverse_target_sequences()
		assert data_iterator.data == reversed_target_data_reference

	def test_ascending_data_sort_source_seq_len(self):
	 	log.info("Testing ascending order sorter by source sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("ascending", "source")
	 	assert data_iterator.data == [[[1],[1,2]],[[1,2],[1]],[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]

	def test_ascending_data_sort_target_seq_len(self):
	 	log.info("Testing ascending order sorter by target sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("ascending", "target")
	 	assert data_iterator.data == [[[1,2],[1]],[[1,2,3],[1,2]],[[1],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]

	def test_ascending_data_sort_source_plus_target_seq_len(self):
	 	log.info("Testing ascending order sorter by sum of source and target sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("ascending", "source+target")
	 	assert data_iterator.data == [[[1],[1,2]],[[1,2],[1]],[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]

	def test_descending_data_sort_source_seq_len(self):
	 	log.info("Testing descending order sorter by source sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("descending", "source")
	 	assert data_iterator.data == [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2]],[[1,2],[1]],[[1],[1,2]]]

	def test_descending_data_sort_target_seq_len(self):
	 	log.info("Testing descending order sorter by target sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("descending", "target")
	 	assert data_iterator.data == [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2]],[[1],[1,2]],[[1,2],[1]]]

	def test_descending_data_sort_source_plus_target_seq_len(self):
	 	log.info("Testing descending order sorter by sum of source and target sequence length.")
	 	data_iterator = DataHandler(list(self.seq2seqdata))
	 	data_iterator.sort("descending", "source+target")
	 	assert data_iterator.data == [[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2]],[[1],[1,2]],[[1,2],[1]]]

	def test_batch_generation_one_epoch_batch_size_one(self):
		log.info("Testing generation of batches for one epoch. Batch size is 1.")
		batched_data = []
		batched_data_reference = [[[[1,2,3],[1,2]]],[[[1,2,3,4,5],[1,2,3,4,5,6,7]]],[[[1],[1,2]]],[[[1,2],[1]]],[[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]]]
		data_iterator = DataHandler(list(self.seq2seqdata_longer))
		for minibatch in data_iterator.batched_data_generator(batch_size = 1, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == batched_data_reference

	def test_batch_generation_one_epoch_batch_size_3(self):
		log.info("Testing generation of batches for one epoch. Batch size is 3.")
		batched_data = []
		batched_data_reference = [[[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]]],[[[1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]]],[[[1,2,3,1,2],[1]]]]
		data_iterator = DataHandler(list(self.seq2seqdata_longer))
		for minibatch in data_iterator.batched_data_generator(batch_size = 3, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == batched_data_reference

	def test_batch_generation_one_epoch_batch_size_onehundred(self):
		log.info("Testing generation of batches for one epoch. Batch size is 100. Basically batch size is greater than available number of training items.")
		batched_data = []
		batched_data_reference = [[[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]],[[1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]]]]
		data_iterator = DataHandler(list(self.seq2seqdata_longer))
		for minibatch in data_iterator.batched_data_generator(batch_size = 100, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == batched_data_reference

	def test_batch_padding_and_mask_generation(self):
		log.info("Testing batch padding and mask generation.")
		in_batch = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]]]
		padded_batch_ref = [[[1,2,3,10,10],[1,2,11,11,11,11,11]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,10,10,10,10],[1,2,11,11,11,11,11]]]
		mask_ref = [[[True,True,True,False,False],[True,True,False,False,False,False,False]],[[True,True,True,True,True],[True,True,True,True,True,True,True]],[[True,False,False,False,False],[True,True,False,False,False,False,False]]]
		padded_batch, mask = pad_batch_and_generate_mask(in_batch, 10, 11)
		assert padded_batch != in_batch
		assert padded_batch == padded_batch_ref
		assert mask == mask_ref

	def test_bucketed_data_generation(self):
		log.info("Testing bucketed data generation.")
		in_data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]],[[1],[1,2]],[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		data_iterator = DataHandler(in_data)
		data_iterator.generate_bucketed_data()
		assert data_iterator.bucketed_data[5] == [[[1,2,3],[1,2]],[[1],[1,2]]]
		assert data_iterator.bucketed_data[10] == [[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]]]
		assert data_iterator.bucketed_data[20] == []
		assert data_iterator.bucketed_data[40] == [[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		assert data_iterator.bucket_distribution == [0.5, 0.75, 0.75, 1.0]

	def test_bucketed_batch_generation_one_epoch_batch_size_one(self):
		log.info("Testing bucketed batch data generation. Batch size is 1. Setting a seed for pseudorandomness of bucket choice.")
		in_data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]],[[1],[1,2]],[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		ref_batched_data = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 12, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [[[1, 2, 3], [1, 2]]], [[[1], [1, 2]]], [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]
		data_iterator = DataHandler(in_data)
		data_iterator.generate_bucketed_data()
		random.seed(1234)
		batched_data = []
		for minibatch in data_iterator.batched_data_generator_for_bucketed_data(batch_size = 1, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == ref_batched_data

	def test_bucketed_batch_generation_one_epoch_batch_size_two(self):
		log.info("Testing bucketed batch data generation. Batch size is 2. Setting a seed for pseudorandomness of bucket choice.")
		in_data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]],[[1],[1,2]],[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		ref_batched_data = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 12, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [[[1, 2, 3], [1, 2]], [[1], [1, 2]]], [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]
		data_iterator = DataHandler(in_data)
		data_iterator.generate_bucketed_data()
		random.seed(1234)
		batched_data = []
		for minibatch in data_iterator.batched_data_generator_for_bucketed_data(batch_size = 2, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == ref_batched_data

	def test_bucketed_batch_generation_one_epoch_batch_size_onehundred(self):
		log.info("Testing bucketed batch data generation. Batch size is 100. Setting a seed for pseudorandomness of bucket choice.")
		in_data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]],[[1],[1,2]],[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		ref_batched_data = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 12, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]], [[[1, 2, 3], [1, 2]], [[1], [1, 2]]], [[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]]
		data_iterator = DataHandler(in_data)
		data_iterator.generate_bucketed_data()
		random.seed(1234)
		batched_data = []
		for minibatch in data_iterator.batched_data_generator_for_bucketed_data(batch_size = 100, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == ref_batched_data
		
	def test_pseudo_random_bucket_selection(self):
		log.info("Testing pseudo random bucket slelection.")
		in_data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7,8,9,10]],[[1],[1,2]],[[1,2,3,4,5,6,7,8,9,11,12,12,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[1,2,3,4,5,6,7,8,9,10]]]
		data_iterator = DataHandler(in_data)
		data_iterator.generate_bucketed_data()
		assert data_iterator.get_random_bucket(force_random = 0.8) == 40
		assert data_iterator.get_random_bucket(force_random = 0.2) == 5

	def test_data_filtering(self):
		log.info("Testing data filtering.")
		data_iterator = DataHandler(self.seq2seqdata)
		data_iterator.filter_data(max_src_len = 1, max_tgt_len = 1)
		assert data_iterator.data == []
		data_iterator = DataHandler(self.seq2seqdata)
		data_iterator.filter_data(max_src_len = 1, max_tgt_len = 2)
		assert data_iterator.data == [[[1],[1,2]]]
		data_iterator = DataHandler(self.seq2seqdata)
		data_iterator.filter_data(max_src_len = 100, max_tgt_len = 100)
		assert data_iterator.data == self.seq2seqdata
		

if __name__ == '__main__':
    unittest.main()