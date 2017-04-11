#!/usr/bin/env python
"""bleu_computer.py: A class and relevant methods to compute the bleu score of translations given references."""
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
import math

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)

logging.basicConfig()
log = logging.getLogger("utilities:nmt:bleu_computer")
log.setLevel(logging.INFO)


class bleu_computer:
	def __init__(self, references, ngram = 4):
		log.info("Initializing the bleu computer.")
		self.ngram = ngram
		self.reference_ngrams, self.reference_length = self.get_ngrams(references)
		

	def get_ngrams(self, sentences):
		log.info("Getting N-grams. Limit is %s-grams." % str(self.ngram))
		total_length = 0
		ngrams_per_sentence = []
		for sentence in sentences:
			ngrams_dict = defaultdict(int)
			sentence_len = len(sentence)
			total_length += sentence_len
			for ngram in range(1, self.ngram + 1):
				for sentence_idx in range(0, sentence_len - ngram + 1):
					ngrams_dict[tuple(sentence[sentence_idx: sentence_idx+ngram])] += 1
			ngrams_per_sentence.append(ngrams_dict)
		return ngrams_per_sentence, total_length

	def compute_ngram_matches(self):
		log.info("Computing n-gram matches.")
		total_ngrams = {x:0 for x in range(1, self.ngram + 1)}
		correct_ngrams = {x:0 for x in range(1, self.ngram + 1)}
		for reference_info, translation_info in zip(self.reference_ngrams, self.trans_ngrams):
			for ngram, translation_freq in translation_info.iteritems():
				n = len(ngram)
				reference_freq = reference_info[ngram]
				total_ngrams[n] += translation_freq
				if ngram in reference_info:
					if reference_freq >= translation_freq:
						correct_ngrams[n] += translation_freq
					else:
						correct_ngrams[n] += reference_freq
		return total_ngrams, correct_ngrams

	def get_full_bleu_info(self):
		log.info("Getting complete BLEU information.")
		final_string = " ".join([" ".join([str(ngram)+"-gram: ", str(self.correct_ngrams[ngram]), "/", str(self.total_ngrams[ngram]), "(", "%2.2f" % (100 * self.ngram_stats[ngram]), "%)"]) for ngram in self.correct_ngrams])
		return ("BLEU score = %2.2f" % (100 * self.bleu)) + "%\n" + "Brevity penalty: " + "%0.4f" % math.exp(self.brevity_penalty) + "\n" + final_string

	def bleu(self, translations):
		log.info("Computing bleu for given translations.")
		self.trans_ngrams, self.trans_len = self.get_ngrams(translations)
		self.total_ngrams, self.correct_ngrams = self.compute_ngram_matches()

		if min(self.correct_ngrams.values()) <= 0:
			return 0
		assert min(self.total_ngrams.values()) >= 0
		assert min(self.total_ngrams.values()) >= min(self.correct_ngrams.values())

		log_brevity_penalty = min(0, 1.0 - float(self.reference_length) / self.trans_len)
		log_average_precision = (1.0 / self.ngram) *(sum(math.log(v) for v in self.correct_ngrams.values()) - sum(math.log(v) for v in self.total_ngrams.values()))
		res = math.exp(log_brevity_penalty + log_average_precision)
		self.bleu = res
		self.ngram_stats = {ngram: 1.0 * self.correct_ngrams[ngram]/self.total_ngrams[ngram] for ngram in self.correct_ngrams}
		self.brevity_penalty = log_brevity_penalty
		return res