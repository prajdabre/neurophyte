#!/usr/bin/env python
"""prepare_seq2seq_data_test.py: Tests for the methods and classes in prepare_seq2seq_data.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
import indexing
from prepare_seq2seq_data import *
import io
import uuid
import logging

logging.basicConfig()
log = logging.getLogger("dataprocessing:nmt:prepare_seq2seq_data")
log.setLevel(logging.INFO)

class TestSeq2SeqDataMakerMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSeq2SeqDataMakerMethods, self).__init__(*args, **kwargs)
        log.info("Initializing the test.")

    def test_get_source_languages(self):
        assert get_source_languages(["fr-en", "fr-de", "de-fr"]) == ["fr", "fr", "de"]

    def test_get_target_languages(self):
        assert get_target_languages(["fr-en", "fr-de", "de-fr"]) == ["en", "de", "fr"]
    
    def test_populate_seq2seq_train_data(self):
        
        populate_seq2seq_data(seq2seqdata, language_pairs, indexed_sources, indexed_targets, type_data = "train")
    
if __name__ == '__main__':
    unittest.main()