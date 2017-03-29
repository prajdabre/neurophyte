#!/usr/bin/env python
"""indexing_test.py: Tests for the methods and classes in indexing.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
import indexing
from indexing import Indexer, generate_dictionary, generate_index
import io
import uuid
import logging

logging.basicConfig()
log = logging.getLogger("dataprocessing:common:indexing_test")
log.setLevel(logging.INFO)

class TestIndexerMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestIndexerMethods, self).__init__(*args, **kwargs)
        log.info("Initializing the test.")

        self.reference_dictionary_full = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'fire': 23, u'pen': 12, 'unique#UNK#euqinu': 2, u'was': 24, u'merely': 25, u'ally': 26, u'that': 27, u'I': 3, 'unique#EOS#euqinu': 1, u'universe': 37, u'by': 29, u'molded': 30, u'a': 31, u'king': 32, u'Apple': 33, u'potato': 34, u'of': 28, u'?': 36, u'Ah': 35, u'the': 7, u'think': 38}
        self.reference_indexed_file_full = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 31, 12, 5, 1], [0, 35, 19, 33, 21, 12, 5, 1], [0, 3, 6, 23, 4, 3, 6, 20, 4, 3, 6, 34, 4, 3, 6, 7, 32, 28, 7, 37, 5, 1], [0, 10, 38, 27, 7, 11, 14, 18, 26, 36, 10, 25, 22, 7, 11, 4, 3, 24, 16, 17, 8, 4, 30, 29, 8, 5, 1]]
        self.reference_word_to_line_map_full = {3: set([0, 1, 3, 4]), 4: set([3, 4]), 5: set([0, 1, 2, 3, 4]), 6: set([3]), 7: set([3, 4]), 8: set([4]), 9: set([0, 1]), 10: set([4]), 11: set([4]), 12: set([1, 2]), 13: set([0]), 14: set([4]), 15: set([0]), 16: set([4]), 17: set([4]), 18: set([4]), 19: set([2]), 20: set([3]), 21: set([2]), 22: set([4]), 23: set([3]), 24: set([4]), 25: set([4]), 26: set([4]), 27: set([4]), 28: set([3]), 29: set([4]), 30: set([4]), 31: set([1]), 32: set([3]), 33: set([2]), 34: set([3]), 35: set([2]), 36: set([4]), 37: set([3]), 38: set([4])}
        self.reference_dictionary_top_20 = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'pen': 12, 'unique#UNK#euqinu': 2, u'I': 3, 'unique#EOS#euqinu': 1, u'the': 7}
        self.reference_indexed_file_top_20 = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 2, 12, 5, 1], [0, 2, 19, 2, 21, 12, 5, 1], [0, 3, 6, 2, 4, 3, 6, 20, 4, 3, 6, 2, 4, 3, 6, 7, 2, 2, 7, 2, 5, 1], [0, 10, 2, 2, 7, 11, 14, 18, 2, 2, 10, 2, 22, 7, 11, 4, 3, 2, 16, 17, 8, 4, 2, 2, 8, 5, 1]]
        self.reference_word_to_line_map_top_20 = {2: set([1, 2, 3, 4]), 3: set([0, 1, 3, 4]), 4: set([3, 4]), 5: set([0, 1, 2, 3, 4]), 6: set([3]), 7: set([3, 4]), 8: set([4]), 9: set([0, 1]), 10: set([4]), 11: set([4]), 12: set([1, 2]), 13: set([0]), 14: set([4]), 15: set([0]), 16: set([4]), 17: set([4]), 18: set([4]), 19: set([2]), 20: set([3]), 21: set([2]), 22: set([4])}
        self.reference_line_to_sequence_full = [0, 3, 9, 31, 12, 5, 1]
        self.reference_line_to_sequence_top_20 = [0, 3, 9, 2, 12, 5, 1]
        self.unique_filename = str(uuid.uuid4())
        f = io.open("/tmp/" + self.unique_filename, "w", encoding = "utf-8")
        f.write(u"I have an apple .\n")
        f.write(u"I have a pen .\n")
        f.write(u"Ah ! Apple - pen .\n")
        f.write(u"I am fire , I am death , I am potato , I am the king of the universe .\n")
        f.write(u"You think that the darkness is your ally ? You merely adopted the darkness , I was born in it , molded by it .\n")
        f.flush()
        f.close()
        
    def test_full_dictionary_generation(self):
        log.info("Testing full dictionary generation.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer)
        assert indexer.dictionary == self.reference_dictionary_full

    def test_top_20_dictionary_generation(self):
        log.info("Testing partial dictionary generation.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer, 20)
        assert indexer.dictionary == self.reference_dictionary_top_20

    def test_full_index_generation(self):
        log.info("Testing index generation for the full dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_map == self.reference_word_to_line_map_full

    def test_top_20_index_generation(self):
        log.info("Testing index generation for the partial dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_map == self.reference_word_to_line_map_top_20

    def test_full_indexed_file_generation(self):
        log.info("Testing indexed file generation for the full dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.indexed_file == self.reference_indexed_file_full

    def test_top_20_indexed_file_generation(self):
        log.info("Testing indexed file generation for the partial dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.indexed_file == self.reference_indexed_file_top_20

    def test_full_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the full dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer)
        assert indexer.convert_line_to_id_sequence(u"I have a pen .\n") == self.reference_line_to_sequence_full

    def test_top_20_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the partial dictionary.")
        indexer = Indexer("/tmp/" + str(self.unique_filename))
        indexer = generate_dictionary(indexer, 20)
        assert indexer.convert_line_to_id_sequence(u"I have a pen .\n") == self.reference_line_to_sequence_top_20
        

if __name__ == '__main__':
    unittest.main()