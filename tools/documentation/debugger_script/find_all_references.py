import os
import pickle

path_file = os.path.dirname(os.path.abspath(__file__))

with open(path_file + "/../../../docs/_build/doctrees/environment.pickle", "rb") as f:
    dat = pickle.load(f)

print(dat.domaindata["std"]["labels"].keys())
