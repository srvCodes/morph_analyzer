import pickle
import gzip


class PickleHandler():
    def __init__(self):
        pass

    @staticmethod
    def pickle_dumper(obj, name):
        pickle.dump(obj, gzip.open('resources/' + name + '.gzip', 'wb'))

    @staticmethod
    def pickle_loader(name):
        obj = pickle.load(gzip.open('resources/' + name + '.gzip', 'rb'))
        return obj

