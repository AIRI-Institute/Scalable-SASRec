from random import randrange
import numpy as np
from numba import njit


@njit(fastmath=True)
def sample_unseen(sample_size, sampler_state, remaining, result):
    """
    Sample a desired number of integers from a range (starting from zero)
    excluding black-listed elements defined in sample state. Used with in
    conjunction with `prime_sample_state` method, which initializes state.
    Inspired by Fischer-Yates shuffle.
    """
    # gradually sample from the decreased size range
    for k in range(sample_size):
        # i = random_state.randint(remaining)
        i = randrange(remaining)
        result[k] = sampler_state.get(i, i)
        remaining -= 1
        sampler_state[i] = sampler_state.get(remaining, remaining)
        sampler_state.pop(remaining, -1)


@njit(fastmath=True)
def prime_sampler_state(n, exclude):
    """
    Initialize state to be used in `sample_unseen_items`.
    Ensures seen items are never sampled by placing them
    outside of sampling region.
    """
    # initialize typed numba dicts
    state = {n: n}; state.pop(n)
    track = {n: n}; track.pop(n)

    n_pos = n - len(state) - 1
    # reindex excluded items, placing them in the end
    for i, item in enumerate(exclude):
        pos = n_pos - i
        x = track.get(item, item)
        t = state.get(pos, pos)
        state[x] = t
        track[t] = x
        state.pop(pos, n)
        track.pop(item, n)
    return state
