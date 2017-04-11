#!/usr/bin/env python
"""bleu_computer_test.py: Tests for the methods and classes in bleu_computer.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
from bleu_computer import *
import io
import uuid
import logging
import numpy as np

logging.basicConfig()
log = logging.getLogger("utilities:nmt:bleu_computer_test")
log.setLevel(logging.INFO)

class TestDataIterators(unittest.TestCase):
	def __init__(self, *args, **kwargs):
		super(TestDataIterators, self).__init__(*args, **kwargs)
		log.info("Initializing the test.")
		self.references = [[1,2,3,4,5,6,7],[3,4,5,6,7,8,9],[5,6,7,8,9,1,1,2]]
		self.translations_reversed = [[7,6,5,4,3,2,1],[9,8,7,6,5,4,3],[2,1,1,9,8,7,6,5]]
		self.translations_random = [[1,3,4,5,6],[2,3,4,7,8,9],[5,6,7,8,9,1,2,2]]
		self.reference_ngrams = [{(5, 6): 1, (2, 3, 4, 5): 1, (1, 2): 1, (5, 6, 7): 1, (6, 7): 1, (1,): 1, (3,): 1, (5,): 1, (7,): 1, (3, 4, 5, 6): 1, (4, 5): 1, (1, 2, 3): 1, (2, 3): 1, (4, 5, 6): 1, (4, 5, 6, 7): 1, (1, 2, 3, 4): 1, (2,): 1, (4,): 1, (6,): 1, (3, 4, 5): 1, (2, 3, 4): 1, (3, 4): 1}, {(5, 6): 1, (9,): 1, (8, 9): 1, (5, 6, 7, 8): 1, (5, 6, 7): 1, (6, 7): 1, (3,): 1, (5,): 1, (7,): 1, (3, 4, 5, 6): 1, (4, 5): 1, (8,): 1, (7, 8, 9): 1, (6, 7, 8): 1, (4, 5, 6): 1, (4, 5, 6, 7): 1, (3, 4): 1, (4,): 1, (6, 7, 8, 9): 1, (6,): 1, (3, 4, 5): 1, (7, 8): 1}, {(9, 1): 1, (7, 8, 9, 1): 1, (5, 6): 1, (9,): 1, (8, 9): 1, (8, 9, 1): 1, (5, 6, 7, 8): 1, (1, 2): 1, (5, 6, 7): 1, (6, 7): 1, (1,): 2, (9, 1, 1): 1, (9, 1, 1, 2): 1, (5,): 1, (7,): 1, (1, 1): 1, (8,): 1, (7, 8, 9): 1, (6, 7, 8): 1, (2,): 1, (1, 1, 2): 1, (6, 7, 8, 9): 1, (6,): 1, (8, 9, 1, 1): 1, (7, 8): 1}]

	def test_ngrams_computer(self):
		log.info("Testing n-grams computer.")
		bc = bleu_computer(self.references)
		assert bc.reference_ngrams == self.reference_ngrams

	def test_bleu_all_matches(self):
		log.info("Testing bleu score for a perfect match to references.")
		bc = bleu_computer(self.references)
		bleu = bc.bleu(self.references)
		assert bleu == 1.0

	def test_bleu_reversed_reference_translations(self):
		log.info("Testing bleu score for translations which are exact reverses of the references but with no 4-gram matches.")
		bc = bleu_computer(self.references)
		bleu = bc.bleu(self.translations_reversed)
		assert bc.correct_ngrams[1] == 22
		assert bc.correct_ngrams[2] == 1
		assert bc.correct_ngrams[3] == 0
		assert bc.correct_ngrams[4] == 0
		assert bleu == 0.0

	def test_bleu_random(self):
		log.info("Testing bleu score for arbitrary matches.")
		bc = bleu_computer(self.references)
		bleu = bc.bleu(self.translations_random)
		assert bc.correct_ngrams[1] == 17
		assert bc.correct_ngrams[2] == 12
		assert bc.correct_ngrams[3] == 7
		assert bc.correct_ngrams[4] == 4
		assert float("%.4f" % bleu) == 0.5265

	def test_full_bleu_info(self):
		log.info("Testing full bleu information generation.")
		bc = bleu_computer(self.references)
		bleu = bc.bleu(self.translations_random)
		assert bc.get_full_bleu_info() == "BLEU score = 52.65%\nBrevity penalty: 0.8539\n1-gram:  17 / 19 ( 89.47 %) 2-gram:  12 / 16 ( 75.00 %) 3-gram:  7 / 13 ( 53.85 %) 4-gram:  4 / 10 ( 40.00 %)"