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

	def random_shuffle(self):
		log.info("Randomly shuffling the input data.")
		self.data = random.shuffle(self.data)
		log.info("Random shuffling complete.")

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