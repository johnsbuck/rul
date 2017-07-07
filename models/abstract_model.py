from abc import ABCMeta
from abc import abstractmethod


class AbstractModel(object):
    """

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, x, y):
        """

        Args:
            x:
            y:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def accuracy(self, x, y):
        """

        Args:
            x:
            y:

        Returns:

        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x):
        """

        Args:
            x:

        Returns:

        """
        raise NotImplementedError
