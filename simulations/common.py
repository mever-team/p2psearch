from random import seed as rnd_seed
from numpy.random import seed as np_rnd_seed


def set_seed(seed):
    """
    Sets the simulation seed for all random number generators.
    """
    rnd_seed(seed)
    np_rnd_seed(seed)
