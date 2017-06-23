#!/usr/bin/env python
"""attentions.py: Classes and relevant methods for various attentions for NMT."""
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
import tensorflow as tf
