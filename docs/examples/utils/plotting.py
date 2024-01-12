#!/usr/bin/env python3

import matplotlib.pyplot as plt

def set_two_y_axes_zeros_equal(ax1: plt.Axes, ax2: plt.Axes):
    """Changes ylims so that the zeros on two axes to occur at the same point.

    Changes the limits of ax2.

    It's intended for a figure with two y-axes on the same plot (using
    plt.twinx()) but will probably work in other contexts

    based on
    https://stackoverflow.com/questions/27135162/guaranteeing-0-at-same-level-on-left-and-right-y-axes-python-matplotlib

    """
    min_left, max_left = ax1.get_ylim()
    min_right, max_right = ax2.get_ylim()
    print(min_right,  max_right)

    ratio_left = abs(min_left)/(max_left+abs(min_left))
    ratio_right = abs(min_right)/(max_right+abs(min_right))

    if ratio_left <= ratio_right:
        max_right = min_right * max_left / min_left
    else:
        min_right = max_right * min_left / max_left
    ax2.set_ylim(min_right, max_right)
