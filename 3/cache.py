import h5py
import numpy as np

_file = h5py.File("out/cache.hdf5")

class Cache:
    def __init__(self, namespace, version, use = True):
        self.ns = namespace
        self.version = version
        self.use = use

    def fetch(self, key, f):
        #if params:
        #    params_key = map(lambda p: str(p).lower(), params)
        #else:
        #    params_key = 'no_params'

        k = "{}/{}/{}".format(self.ns, self.version, key)
        if self.use and k in _file:
            print("Using cached {}".format(k))
            return _file[k][:]
        else:
            val = f()
            if k in _file:
                del _file[k]
            _file[k] = val
            return val
