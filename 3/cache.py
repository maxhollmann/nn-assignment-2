import h5py
import numpy as np
import os


_file = h5py.File("cache.hdf5")

class Cache:
    def __init__(self, namespace, version, use = True):
        self.ns = namespace
        self.version = version
        self.use = use

    def fetch(self, key, f):
        k = "{}/{}/{}".format(self.ns, self.version, key)
        if self.use and k in _file:
            print("Using cached {}".format(k))
            return _file[k][:]
        else:
            print("No cache for {}".format(k))
            val = f()
            if k in _file:
                del _file[k]
            _file[k] = val
            return val
