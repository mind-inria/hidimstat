import inspect
import os
import sys


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    try:
        module_name = info["module"]
        fullname = info["fullname"]
        submod = sys.modules.get(module_name)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(__file__))

        # Adjust if project inside src folder
        if fn.startswith("src"):
            fn = fn[len("src/") :]

        return f"https://github.com/mind-inria/hidimstat/blob/main/src/{fn}#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        return None
