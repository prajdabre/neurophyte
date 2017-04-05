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

	def test_batch_generation_one_epoch_batch_size_one(self):
		log.info("Testing generation of batches for one epoch. Batch size is 100. Basically batch size is greater than available number of training items.")
		batched_data = []
		batched_data_reference = [[[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1],[1,2]],[[1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]],[[1,2,3,1,2],[1]]]]
		data_iterator = DataHandler(list(self.seq2seqdata_longer))
		for minibatch in data_iterator.batched_data_generator(batch_size = 100, num_epochs = 1, sorted_batches = False):
			batched_data.append(minibatch)
		assert batched_data == batched_data_reference

if __name__ == '__main__':
    unittest.main()