import os
import pickle

# This is tool is based on: https://stackoverflow.com/questions/26666859/how-do-i-print-all-labels-that-are-defined-in-a-sphinx-project
# It's for getting all the labels generated during the documentation.
# It's useful when we check the reason for undefined label warnings.

path_file = os.path.dirname(os.path.abspath(__file__))

with open(path_file + "/../../../docs/_build/doctrees/environment.pickle", "rb") as f:
    dat = pickle.load(f)

print(dat.domaindata["std"]["labels"].keys())
