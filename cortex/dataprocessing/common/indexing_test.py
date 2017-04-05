#!/usr/bin/env python
"""indexing_test.py: Tests for the methods and classes in indexing.py."""
__author__ = "Raj Dabre"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "prajdabre@gmail.com"
__status__ = "Development"


import unittest
import indexing
from indexing import MultiIndexer, generate_dictionary, generate_index
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

        self.reference_dictionary_full = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, u'unique#PADDING#euqinu': 39, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'fire': 23, u'pen': 12, 'unique#UNK#euqinu': 2, u'was': 24, u'merely': 25, u'ally': 26, u'that': 27, u'I': 3, 'unique#EOS#euqinu': 1, u'universe': 37, u'by': 29, u'molded': 30, u'a': 31, u'king': 32, u'Apple': 33, u'potato': 34, u'of': 28, u'?': 36, u'Ah': 35, u'the': 7, u'think': 38}
        self.reference_indexed_file_full = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 31, 12, 5, 1], [0, 35, 19, 33, 21, 12, 5, 1], [0, 3, 6, 23, 4, 3, 6, 20, 4, 3, 6, 34, 4, 3, 6, 7, 32, 28, 7, 37, 5, 1], [0, 10, 38, 27, 7, 11, 14, 18, 26, 36, 10, 25, 22, 7, 11, 4, 3, 24, 16, 17, 8, 4, 30, 29, 8, 5, 1]]
        self.reference_word_to_line_map_full = {3: set([0, 1, 3, 4]), 4: set([3, 4]), 5: set([0, 1, 2, 3, 4]), 6: set([3]), 7: set([3, 4]), 8: set([4]), 9: set([0, 1]), 10: set([4]), 11: set([4]), 12: set([1, 2]), 13: set([0]), 14: set([4]), 15: set([0]), 16: set([4]), 17: set([4]), 18: set([4]), 19: set([2]), 20: set([3]), 21: set([2]), 22: set([4]), 23: set([3]), 24: set([4]), 25: set([4]), 26: set([4]), 27: set([4]), 28: set([3]), 29: set([4]), 30: set([4]), 31: set([1]), 32: set([3]), 33: set([2]), 34: set([3]), 35: set([2]), 36: set([4]), 37: set([3]), 38: set([4])}
        self.reference_dictionary_top_20 = {u'apple': 13, u'is': 14, u'am': 6, u'it': 8, u'an': 15, u'born': 16, u'have': 9, u'in': 17, u'You': 10, u'your': 18, u'!': 19, u'unique#PADDING#euqinu': 23, 'unique#BOS#euqinu': 0, u'death': 20, u'darkness': 11, u'-': 21, u',': 4, u'adopted': 22, u'.': 5, u'pen': 12, 'unique#UNK#euqinu': 2, u'I': 3, 'unique#EOS#euqinu': 1, u'the': 7}
        self.reference_indexed_file_top_20 = [[0, 3, 9, 15, 13, 5, 1], [0, 3, 9, 2, 12, 5, 1], [0, 2, 19, 2, 21, 12, 5, 1], [0, 3, 6, 2, 4, 3, 6, 20, 4, 3, 6, 2, 4, 3, 6, 7, 2, 2, 7, 2, 5, 1], [0, 10, 2, 2, 7, 11, 14, 18, 2, 2, 10, 2, 22, 7, 11, 4, 3, 2, 16, 17, 8, 4, 2, 2, 8, 5, 1]]
        self.reference_word_to_line_map_top_20 = {2: set([1, 2, 3, 4]), 3: set([0, 1, 3, 4]), 4: set([3, 4]), 5: set([0, 1, 2, 3, 4]), 6: set([3]), 7: set([3, 4]), 8: set([4]), 9: set([0, 1]), 10: set([4]), 11: set([4]), 12: set([1, 2]), 13: set([0]), 14: set([4]), 15: set([0]), 16: set([4]), 17: set([4]), 18: set([4]), 19: set([2]), 20: set([3]), 21: set([2]), 22: set([4])}

        self.reference_dictionary_multi_full = {u'and': 11, u'apple': 24, u'For': 25, u'people': 26, u'is': 27, u'?': 68, u'am': 7, u'it': 14, u'an': 28, u'How': 23, u'are': 30, u'have': 15, u'in': 16, u'You': 17, u'In': 71, u'your': 31, u'!': 12, 'unique#BOS#euqinu': 0, u'death': 33, u'from': 34, u'darkness': 8, u'would': 35, u'there': 38, u'-': 39, u',': 6, u'adopted': 40, u'.': 4, u'source': 36, u'pen': 18, u'triggered': 50, u'get': 43, 'unique#UNK#euqinu': 2, u'memes': 9, u'was': 19, u'energy': 42, u'ally': 46, u'purposelessness': 47, u'a': 59, u'them': 48, u'affected': 49, u'fire': 41, u'sprang': 32, u'I': 5, 'unique#EOS#euqinu': 1, u'forth': 55, u'universe': 22, u'Apple': 61, u'were': 52, u'respite': 53, u'they': 54, u'not': 37, u'The': 56, u'beginning': 57, u'by': 13, u'molded': 58, u'And': 20, u'that': 51, u'for': 44, u'born': 29, u'be': 63, u'Ah': 21, u'potato': 62, u'life': 64, u'awe': 65, u'this': 66, u'ones': 67, u'merely': 45, u'of': 10, u'the': 3, u'think': 69, u'glorious': 70, u'king': 60, u'unique#PADDING#euqinu': 72}
        self.reference_indexed_files_multi_full = [[[0, 5, 15, 28, 24, 4, 1], [0, 5, 15, 59, 18, 4, 1], [0, 21, 12, 61, 39, 18, 4, 1], [0, 5, 7, 41, 6, 5, 7, 33, 6, 5, 7, 62, 6, 5, 7, 3, 60, 10, 3, 22, 4, 1], [0, 17, 69, 51, 3, 8, 27, 31, 46, 68, 17, 45, 40, 3, 8, 6, 5, 19, 29, 16, 14, 6, 58, 13, 14, 4, 1]], [[0, 71, 3, 57, 38, 19, 8, 4, 1], [0, 20, 34, 3, 8, 32, 55, 3, 9, 4, 1], [0, 21, 12, 56, 9, 12, 23, 70, 54, 52, 4, 1], [0, 20, 26, 35, 63, 16, 65, 10, 3, 9, 11, 37, 43, 50, 13, 48, 4, 1], [0, 25, 9, 30, 3, 36, 10, 42, 11, 53, 44, 3, 67, 49, 13, 3, 47, 10, 66, 64, 11, 22, 4, 1]]]
        self.reference_word_to_line_maps_multi_full = [{3: set([3, 4]), 4: set([0, 1, 2, 3, 4]), 5: set([0, 1, 3, 4]), 6: set([3, 4]), 7: set([3]), 8: set([4]), 10: set([3]), 12: set([2]), 13: set([4]), 14: set([4]), 15: set([0, 1]), 16: set([4]), 17: set([4]), 18: set([1, 2]), 19: set([4]), 21: set([2]), 22: set([3]), 24: set([0]), 27: set([4]), 28: set([0]), 29: set([4]), 31: set([4]), 33: set([3]), 39: set([2]), 40: set([4]), 41: set([3]), 45: set([4]), 46: set([4]), 51: set([4]), 58: set([4]), 59: set([1]), 60: set([3]), 61: set([2]), 62: set([3]), 68: set([4]), 69: set([4])}, {3: set([0, 1, 3, 4]), 4: set([0, 1, 2, 3, 4]), 8: set([0, 1]), 9: set([1, 2, 3, 4]), 10: set([3, 4]), 11: set([3, 4]), 12: set([2]), 13: set([3, 4]), 16: set([3]), 19: set([0]), 20: set([1, 3]), 21: set([2]), 22: set([4]), 23: set([2]), 25: set([4]), 26: set([3]), 30: set([4]), 32: set([1]), 34: set([1]), 35: set([3]), 36: set([4]), 37: set([3]), 38: set([0]), 42: set([4]), 43: set([3]), 44: set([4]), 47: set([4]), 48: set([3]), 49: set([4]), 50: set([3]), 52: set([2]), 53: set([4]), 54: set([2]), 55: set([1]), 56: set([2]), 57: set([0]), 63: set([3]), 64: set([4]), 65: set([3]), 66: set([4]), 67: set([4]), 70: set([2]), 71: set([0])}]

        self.reference_dictionary_multi_top_20 = {u'and': 11, u'am': 7, u'it': 14, u'have': 15, u'in': 16, u'You': 17, u'!': 12, 'unique#BOS#euqinu': 0, u'darkness': 8, u',': 6, u'.': 4, u'pen': 18, 'unique#UNK#euqinu': 2, u'memes': 9, u'was': 19, u'I': 5, 'unique#EOS#euqinu': 1, u'universe': 22, u'by': 13, u'And': 20, u'Ah': 21, u'of': 10, u'the': 3, u'unique#PADDING#euqinu': 23}
        self.reference_indexed_files_multi_top_20 = [[[0, 5, 15, 2, 2, 4, 1], [0, 5, 15, 2, 18, 4, 1], [0, 21, 12, 2, 2, 18, 4, 1], [0, 5, 7, 2, 6, 5, 7, 2, 6, 5, 7, 2, 6, 5, 7, 3, 2, 10, 3, 22, 4, 1], [0, 17, 2, 2, 3, 8, 2, 2, 2, 2, 17, 2, 2, 3, 8, 6, 5, 19, 2, 16, 14, 6, 2, 13, 14, 4, 1]], [[0, 2, 3, 2, 2, 19, 8, 4, 1], [0, 20, 2, 3, 8, 2, 2, 3, 9, 4, 1], [0, 21, 12, 2, 9, 12, 2, 2, 2, 2, 4, 1], [0, 20, 2, 2, 2, 16, 2, 10, 3, 9, 11, 2, 2, 2, 13, 2, 4, 1], [0, 2, 9, 2, 3, 2, 10, 2, 11, 2, 2, 3, 2, 2, 13, 3, 2, 10, 2, 2, 11, 22, 4, 1]]]
        self.reference_word_to_line_maps_multi_top_20 = [{2: set([0, 1, 2, 3, 4]), 3: set([3, 4]), 4: set([0, 1, 2, 3, 4]), 5: set([0, 1, 3, 4]), 6: set([3, 4]), 7: set([3]), 8: set([4]), 10: set([3]), 12: set([2]), 13: set([4]), 14: set([4]), 15: set([0, 1]), 16: set([4]), 17: set([4]), 18: set([1, 2]), 19: set([4]), 21: set([2]), 22: set([3])}, {2: set([0, 1, 2, 3, 4]), 3: set([0, 1, 3, 4]), 4: set([0, 1, 2, 3, 4]), 8: set([0, 1]), 9: set([1, 2, 3, 4]), 10: set([3, 4]), 11: set([3, 4]), 12: set([2]), 13: set([3, 4]), 16: set([3]), 19: set([0]), 20: set([1, 3]), 21: set([2]), 22: set([4])}]

        self.reference_line_to_sequence_full = [0, 3, 9, 31, 12, 5, 1]
        self.reference_line_to_sequence_top_20 = [0, 3, 9, 2, 12, 5, 1]
        self.reference_line_to_sequence_multi_full = [0, 20, 34, 3, 8, 32, 55, 3, 9, 4, 1]
        self.reference_line_to_sequence_multi_top_20 = [0, 20, 2, 3, 8, 2, 2, 3, 9, 4, 1]

        self.unique_filename = [str(uuid.uuid4()), str(uuid.uuid4())]
        f = io.open("/tmp/" + self.unique_filename[0], "w", encoding = "utf-8")
        f.write(u"I have an apple .\n")
        f.write(u"I have a pen .\n")
        f.write(u"Ah ! Apple - pen .\n")
        f.write(u"I am fire , I am death , I am potato , I am the king of the universe .\n")
        f.write(u"You think that the darkness is your ally ? You merely adopted the darkness , I was born in it , molded by it .\n")
        f.flush()
        f.close()

        f = io.open("/tmp/" + self.unique_filename[1], "w", encoding = "utf-8")
        f.write(u"In the beginning there was darkness .\n")
        f.write(u"And from the darkness sprang forth the memes .\n")
        f.write(u"Ah ! The memes ! How glorious they were .\n")
        f.write(u"And people would be in awe of the memes and not get triggered by them .\n")
        f.write(u"For memes are the source of energy and respite for the ones affected by the purposelessness of this life and universe .\n")
        f.flush()
        f.close()
        
    def test_full_dictionary_generation(self):
        log.info("Testing full dictionary generation.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer)
        assert indexer.dictionary == self.reference_dictionary_full

    def test_top_20_dictionary_generation(self):
        log.info("Testing partial dictionary generation.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer, 20)
        assert indexer.dictionary == self.reference_dictionary_top_20

    def test_full_index_generation(self):
        log.info("Testing index generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_maps[0] == self.reference_word_to_line_map_full

    def test_top_20_index_generation(self):
        log.info("Testing index generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_maps[0] == self.reference_word_to_line_map_top_20

    def test_full_indexed_file_generation(self):
        log.info("Testing indexed file generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.indexed_files[0] == self.reference_indexed_file_full

    def test_top_20_indexed_file_generation(self):
        log.info("Testing indexed file generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.indexed_files[0] == self.reference_indexed_file_top_20

    def test_full_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer)
        assert indexer.convert_line_to_id_sequence(u"I have a pen .\n")[0] == self.reference_line_to_sequence_full

    def test_top_20_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0]])
        indexer = generate_dictionary(indexer, 20)
        assert indexer.convert_line_to_id_sequence(u"I have a pen .\n")[0] == self.reference_line_to_sequence_top_20
    
    def test_full_dictionary_generation_multi(self):
        log.info("Testing full dictionary generation.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer)
        assert indexer.dictionary == self.reference_dictionary_multi_full

    def test_top_20_dictionary_generation_multi(self):
        log.info("Testing partial dictionary generation.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer, 20)
        assert indexer.dictionary == self.reference_dictionary_multi_top_20

    def test_full_index_generation_multi(self):
        log.info("Testing index generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_maps == self.reference_word_to_line_maps_multi_full

    def test_top_20_index_generation_multi(self):
        log.info("Testing index generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.word_to_line_maps == self.reference_word_to_line_maps_multi_top_20

    def test_full_indexed_file_generation_multi(self):
        log.info("Testing indexed file generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer)
        indexer = generate_index(indexer)
        assert indexer.indexed_files == self.reference_indexed_files_multi_full

    def test_top_20_indexed_file_generation_multi(self):
        log.info("Testing indexed file generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer, 20)
        indexer = generate_index(indexer)
        assert indexer.indexed_files == self.reference_indexed_files_multi_top_20

    def test_full_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the full dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer)
        assert indexer.convert_line_to_id_sequence(u"And from the darkness sprang forth the memes .\n")[0] == self.reference_line_to_sequence_multi_full

    def test_top_20_dictionary_line_to_sequence(self):
        log.info("Testing line to sequence id generation for the partial dictionary.")
        indexer = MultiIndexer(["/tmp/" + self.unique_filename[0], "/tmp/" + self.unique_filename[1]])
        indexer = generate_dictionary(indexer, 20)
        assert indexer.convert_line_to_id_sequence(u"And from the darkness sprang forth the memes .\n")[0] == self.reference_line_to_sequence_multi_top_20

if __name__ == '__main__':
    unittest.main()