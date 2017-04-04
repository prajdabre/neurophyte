#!/usr/bin/env python
"""prepare_seq2seq_data_test.py: Tests for the methods and classes in prepare_seq2seq_data.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
from cortex.dataprocessing.common.indexing import *
from prepare_seq2seq_data import *
import io
import uuid
import logging

logging.basicConfig()
log = logging.getLogger("dataprocessing:nmt:prepare_seq2seq_data_test")
log.setLevel(logging.INFO)

class TestSeq2SeqDataMakerMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSeq2SeqDataMakerMethods, self).__init__(*args, **kwargs)
        log.info("Initializing the test.")

        self.reference_dictionary = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'fire': 23, u'pen': 12, 'unique#UNK#euqinu': 2, u'was': 24, u'merely': 25, u'ally': 26, u'that': 27, u'I': 3, 'unique#EOS#euqinu': 1, u'universe': 37, u'by': 29, u'molded': 30, u'a': 31, u'king': 32, u'Apple': 33, u'potato': 34, u'of': 28, u'?': 36, u'Ah': 35, u'the': 7, u'think': 38}
        self.reference_indexed_file = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 31, 12, 5, 1], [0, 35, 19, 33, 21, 12, 5, 1], [0, 3, 6, 23, 4, 3, 6, 20, 4, 3, 6, 34, 4, 3, 6, 7, 32, 28, 7, 37, 5, 1], [0, 10, 38, 27, 7, 11, 14, 18, 26, 36, 10, 25, 22, 7, 11, 4, 3, 24, 16, 17, 8, 4, 30, 29, 8, 5, 1]]
        self.reference_dictionary_top_20 = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'pen': 12, 'unique#UNK#euqinu': 2, u'I': 3, 'unique#EOS#euqinu': 1, u'the': 7}
        self.reference_indexed_file_top_20 = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 2, 12, 5, 1], [0, 2, 19, 2, 21, 12, 5, 1], [0, 3, 6, 2, 4, 3, 6, 20, 4, 3, 6, 2, 4, 3, 6, 7, 2, 2, 7, 2, 5, 1], [0, 10, 2, 2, 7, 11, 14, 18, 2, 2, 10, 2, 22, 7, 11, 4, 3, 2, 16, 17, 8, 4, 2, 2, 8, 5, 1]]
        
        self.unique_filename = str(uuid.uuid4())
        f = io.open("/tmp/" + self.unique_filename, "w", encoding = "utf-8")
        f.write(u"I have an apple .\n")
        f.write(u"I have a pen .\n")
        f.write(u"Ah ! Apple - pen .\n")
        f.write(u"I am fire , I am death , I am potato , I am the king of the universe .\n")
        f.write(u"You think that the darkness is your ally ? You merely adopted the darkness , I was born in it , molded by it .\n")
        f.flush()
        f.close()

    def test_get_source_languages(self):
        assert get_source_languages(["fr-en", "fr-de", "de-fr"]) == ["fr", "fr", "de"]

    def test_get_target_languages(self):
        assert get_target_languages(["fr-en", "fr-de", "de-fr"]) == ["en", "de", "fr"]
    
    def test_populate_seq2seq_train_data(self):
        log.info("Testing seq2seq data population.")
        dummy_args = {}
        seq2seqdata = Seq2SeqData(dummy_args)
        # Dummy data begins
        language_pairs = ["fr-en", "en-fr", "fr-fr"]
        indexed_sources = {}
        indexed_targets = {}
        indexed_sources["fr"] = MultiIndexer(["f1", "f2"])
        indexed_sources["en"] = MultiIndexer(["f1"])
        indexed_targets["en"] = MultiIndexer(["f1"])
        indexed_targets["fr"] = MultiIndexer(["f1", "f2"])
        indexed_sources["fr"].dictionary = {"a":1, "b":2, "c":3}
        indexed_sources["en"].dictionary = {"x":1, "y":2, "z":3}
        indexed_targets["fr"].dictionary = {"u":1, "v":2, "w":3}
        indexed_targets["en"].dictionary = {"l":1, "m":2, "n":3}
        indexed_sources["fr"].indexed_files = [[[1,2,3],[1,2,2,3]],[[2,3,4],[1,4,5],[4,5,6,6,7]]]
        indexed_sources["en"].indexed_files = [[[3,3,5],[1,1,5],[4,5,1,6,7]]]
        indexed_targets["en"].indexed_files = [[[3,7,1],[7,1,2,3,4]]]
        indexed_targets["fr"].indexed_files = [[[3,1,5],[1,2,5],[4,5,1,4,6]],[[1,2,1],[2,1,2],[3,3,3]]]
        # Dummy data ends
        populate_seq2seq_data(seq2seqdata, language_pairs, indexed_sources, indexed_targets, type_data = "train")
        log.info("Testing dictionary population.")
        assert seq2seqdata.source_dictionaries["fr"] == {"a":1, "b":2, "c":3}
        assert seq2seqdata.source_dictionaries["en"] == {"x":1, "y":2, "z":3}
        assert seq2seqdata.target_dictionaries["fr"] == {"u":1, "v":2, "w":3}
        assert seq2seqdata.target_dictionaries["en"] == {"l":1, "m":2, "n":3}
        log.info("Testing training data population.")
        assert seq2seqdata.pairwise_training_data["fr-en"] == [[[1,2,3],[3,7,1]],[[1,2,2,3],[7,1,2,3,4]]]
        assert seq2seqdata.pairwise_training_data["en-fr"] == [[[3,3,5],[3,1,5]],[[1,1,5],[1,2,5]],[[4,5,1,6,7],[4,5,1,4,6]]] 
        assert seq2seqdata.pairwise_training_data["fr-fr"] == [[[2,3,4],[1,2,1]],[[1,4,5],[2,1,2]],[[4,5,6,6,7],[3,3,3]]]
    
    def test_populate_seq2seq_indexers(self):
        log.info("Testing seq2seq indexers population.")
        indexed_sources, indexed_targets = populate_indexers(["fr", "en", "fr"], ["en", "fr", "fr"], ["/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename], ["/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename], 20000, 20000)
        indexed_sources_top_20, indexed_targets_top_20 = populate_indexers(["fr", "en", "fr"], ["en", "fr", "fr"], ["/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename], ["/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename, "/tmp/" + self.unique_filename], 20, 20)
        
        log.info("Testing dictionary creation. Full dictionary setting.")
        assert indexed_sources["fr"].dictionary == self.reference_dictionary
        assert indexed_sources["en"].dictionary == self.reference_dictionary
        assert indexed_targets["fr"].dictionary == self.reference_dictionary
        assert indexed_targets["en"].dictionary == self.reference_dictionary
        
        log.info("Testing dictionary creation. Top 20 dictionary setting.")
        assert indexed_sources_top_20["fr"].dictionary == self.reference_dictionary_top_20
        assert indexed_sources_top_20["en"].dictionary == self.reference_dictionary_top_20
        assert indexed_targets_top_20["fr"].dictionary == self.reference_dictionary_top_20
        assert indexed_targets_top_20["en"].dictionary == self.reference_dictionary_top_20
        
        log.info("Testing indexed files creation. Full dictionary setting.")
        assert indexed_sources["fr"].indexed_files == [self.reference_indexed_file, self.reference_indexed_file]
        assert indexed_sources["en"].indexed_files == [self.reference_indexed_file]
        assert indexed_targets["fr"].indexed_files == [self.reference_indexed_file, self.reference_indexed_file]
        assert indexed_targets["en"].indexed_files == [self.reference_indexed_file]

        log.info("Testing indexed files creation. Top 20 dictionary setting.")
        assert indexed_sources_top_20["fr"].indexed_files == [self.reference_indexed_file_top_20, self.reference_indexed_file_top_20]
        assert indexed_sources_top_20["en"].indexed_files == [self.reference_indexed_file_top_20]
        assert indexed_targets_top_20["fr"].indexed_files == [self.reference_indexed_file_top_20, self.reference_indexed_file_top_20]
        assert indexed_targets_top_20["en"].indexed_files == [self.reference_indexed_file_top_20]


if __name__ == '__main__':
    unittest.main()