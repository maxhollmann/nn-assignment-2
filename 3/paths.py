import os

def dir(*dirs):
        dir = os.path.join("out", *map(str, dirs))
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

def f(*dirs, f):
    return os.path.join(dir(*dirs), f)
