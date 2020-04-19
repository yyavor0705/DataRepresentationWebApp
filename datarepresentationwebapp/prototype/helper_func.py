import numpy as np
import matplotlib.pyplot as plt

N_SAMPLE = int(1e5)  # Declare first, as it provides global default value for helper functions


# Helper function
def hist_custom(s, ax=None, n_sample=N_SAMPLE):
    """plot histgram with N_SAMPLE//100 bins ignoring nans"""
    if ax is None:
        ax = plt.gca()
    ax.hist(s, bins=min(N_SAMPLE // 100, 100), density=True, alpha=0.5, color='C0')


# Sampler updated
def normal_custom(m, s, n_sample=N_SAMPLE):
    """ Sampling from a normal distribution"""
    x = np.random.normal(loc=m, scale=s, size=n_sample)
    return x
