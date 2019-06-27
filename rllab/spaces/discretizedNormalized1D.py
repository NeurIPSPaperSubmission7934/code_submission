import numpy as np
from .box import Box
import numpy as np

class DiscretizedNormalized1D(Box):
    """
    a discretized space of a 1 dim continuous space
    """

    def __init__(self, low, high, n, right):
        # create bins for the discretization
        self.bins = np.linspace(low, high, n)
        self.right = right
        self._n = n
        # create bins between 0 and 1 for output
        self.discretized_space = np.linspace(0, 1, n)
        # create 1D-box
        super(DiscretizedNormalized1D, self).__init__(0, 1, (1,))

    def toDiscrete(self, continuousX):
        """
        converts a continuous input to the corresponding discrete element
        """
        # first discretize each individual component of continuousX
        discreteIndex = (np.digitize(continuousX, self.bins, right=self.right))
        # note that digitize returns len(bins) if x is beyond the right bound
        # therefore substract 1 from it to be inside of the discretized space and to return a proper index
        if discreteIndex >= self._n:
            discreteIndex = self._n - 1
        return self.discretized_space[discreteIndex]

    def contains(self, continuousX):
        discreteX = self.toDiscrete(continuousX)
        return super(DiscretizedNormalized1D, self).contains(discreteX)

    def toLinearInd(self, continuousX):
        discreteX = self.toDiscrete(continuousX)
        return super(DiscretizedNormalized1D, self).toLinearInd(discreteX)

    def flatten(self, continuousX):
        discreteX = self.toDiscrete(continuousX)
        return super(DiscretizedNormalized1D, self).flatten(discreteX)

    def flatten_n(self, continuousXs):
        discreteXs = self.toDiscrete(continuousXs)
        return super(DiscretizedNormalized1D, self).flatten_n(discreteXs)

    def unflatten(self, continuousX):
        discreteX = self.toDiscrete(continuousX)
        return super(DiscretizedNormalized1D, self).unflatten(discreteX)

    def unflatten_n(self, continuousXs):
        discreteXs = self.toDiscrete(continuousXs)
        return super(DiscretizedNormalized1D, self).unflatten_n(discreteXs)

    def __eq__(self, other):
        if not isinstance(other, DiscretizedNormalized1D):
            return False
        return super(DiscretizedNormalized1D, self).__eq__(other)

    def __repr__(self):
        return "DiscretizedNormalized1D:"  # remove the ', ' from the last component
