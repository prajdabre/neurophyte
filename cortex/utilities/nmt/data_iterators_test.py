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
        self.data = [[[1,2,3],[1,2]],[[1,2,3,4,5],[1,2,3,4,5,6,7]],[[1,2,3],[1,2,3]],[[1,2],[1]]]

	def test_ascending_data_sort(self):
	 	log.info("Testing ascending order sorter.")
	 	data_iterator = DataHandler(list(self.data))
	 	data_iterator.sort("ascending", "source")
	 	assert data_iterator.data == [[[1,2],[1]],[[1,2,3],[1,2]],[[1,2,3],[1,2,3]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]
	 	data_iterator.sort("ascending", "target")
	 	assert data_iterator.data == [[[1,2],[1]],[[1,2,3],[1,2]],[[1,2,3],[1,2,3]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]
	 	data_iterator.sort("ascending", "source+target")
	 	assert data_iterator.data == [[[1,2],[1]],[[1,2,3],[1,2]],[[1,2,3],[1,2,3]],[[1,2,3,4,5],[1,2,3,4,5,6,7]]]

if __name__ == '__main__':
    unittest.main()