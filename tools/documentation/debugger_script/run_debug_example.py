import codeop

from joblib import Parallel, delayed

compiler = codeop.Compile()


def run_example(path):
    """
    Run the code like sphinx gallery do it

    Parameters
    ----------
    path: string
        path of the example
    """
    # Read the file
    with open(path) as f:
        code = f.read()

    # Compile the code
    compiled_code = compiler(code, "plot.py", "exec")

    # Execute the compiled code
    exec(compiled_code, globals())


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    N_JOBS = sys.argv[2] if len(sys.argv) > 2 else 2
    REPEAT = sys.argv[3] if len(sys.argv) > 3 else 2

    parallel = Parallel(n_jobs=N_JOBS, verbose=10)
    results = parallel(delayed(run_example)(path) for i in range(REPEAT))
