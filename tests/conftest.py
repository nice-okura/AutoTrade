import sys
import os

print("### conftest start ###")
sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../src/"))
sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../tests/"))
print("### conftest end ###")
