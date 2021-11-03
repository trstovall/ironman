
from glob import glob


class Bits(object):
    
    def __init__(self):
        return super().__init__()
    
    def __getitem__(self, slice):
        return slice
    
    def __setitem__(self, slice, value):
        return value
    
    def __getattribute__(self, name):
        return super().__getattribute__(name)
    
    def __setattribute__(self, name, attr):
        return super().__setattribute__(name, attr)

