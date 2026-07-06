# Make the howdy python sources importable from the tests
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "howdy", "src")
sys.path.insert(0, SRC_DIR)
