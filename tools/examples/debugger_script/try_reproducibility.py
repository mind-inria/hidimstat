import os
import sys


from joblib import Parallel, delayed

# add the example for import them
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../examples")


def run_joblib(i):
    import plot_knockoff_aggregation  # include the example to test


# run in parallel the same example for compare result
parallel = Parallel(n_jobs=4)
parallel(delayed(run_joblib)(i) for i in range(6))
