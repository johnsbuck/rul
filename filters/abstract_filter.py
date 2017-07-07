from abc import ABCMeta
from abc import abstractmethod


class AbstractFilter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def filter(self):
        raise NotImplementedError

    def smooth(self):
        raise NotImplementedError