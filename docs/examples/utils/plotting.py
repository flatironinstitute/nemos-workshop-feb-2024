#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import pynapple as nap
import numpy as np
from numpy.typing import NDArray
from matplotlib.animation import FuncAnimation


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


def plot_head_direction_tuning(
        tuning_curves: pd.DataFrame,
        spikes: nap.TsGroup,
        angle: nap.Tsd,
        threshold_hz: int = 1,
        start: float = 8910,
        end: float = 8960,
        cmap_label="hsv",
        figsize=(12, 6)
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:

    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------

    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (unq_angles[-1] - unq_angles[0])
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid((3, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid((3, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig)
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(spike_tsd[sel].t, np.ones(sel.sum()) * i, "|", color=cmap(relative_color_levs[i]), alpha=0.5)
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid((3, n_subplots), loc=(2 + i // n_subplots, i % n_subplots),
                              rowspan=1, colspan=1, fig=fig, projection="polar")
        ax.fill_between(tuning_curves.iloc[:, neu_idx].index, np.zeros(len(tuning_curves)),
                        tuning_curves.iloc[:, neu_idx].values, color=cmap(relative_color_levs[i]), alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_count_history_window(
        counts: nap.Tsd,
        n_shift: int,
        history_window: float,
        bin_size: float,
        start: float,
        ylim: tuple[float, float],
        plot_every: int
):
    """
    Plot the count history rolling window.

    Parameters
    ----------
    counts:
        The spike counts of a neuron.
    n_shift:
        Number of rolling windows to plot.
    history_window:
        Size of the history window in seconds.
    bin_size:
        Bin size of the counts in seconds.
    start:
        Start time for the first plotted window
    ylim:
        y limits for axes.
    plot_every:
        Plot a window series every "plot_every" bins

    Returns
    -------

    """
    interval = nap.IntervalSet(start, start + history_window + bin_size * n_shift * plot_every)
    fig, axs = plt.subplots(n_shift, 1, figsize=(8, 8))
    for shift_bin in range(0, n_shift*plot_every, plot_every):
        ax = axs[shift_bin // plot_every]

        shift_sec = shift_bin * bin_size
        # select the first bin after one sec
        input_interval = nap.IntervalSet(
            start=interval["start"][0] + shift_sec,
            end=history_window + interval["start"][0] + shift_sec - 0.001
        )
        predicted_interval = nap.IntervalSet(
            start=history_window + interval["start"][0] + shift_sec,
            end=history_window + interval["start"][0] + bin_size + shift_sec
        )

        ax.step(counts.restrict(interval).t, counts.restrict(interval).d, where="post")

        ax.axvspan(
            input_interval["start"][0],
            input_interval["end"][0], *ylim, alpha=0.4, color="orange")
        ax.axvspan(
            predicted_interval["start"][0],
            predicted_interval["end"][0], *ylim, alpha=0.4, color="tomato"
        )

        plt.ylim(ylim)
        if shift_bin == 0:
            ax.set_title("Spike Count Time Series")
        elif shift_bin == n_shift - 1:
            ax.set_xlabel("Time (sec)")
        if shift_bin != n_shift - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        if shift_bin == 0:
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color("tomato")
                ax.spines[spine].set_linewidth(2)
        else:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_features(
        input_feature: NDArray,
        neuron_id: int,
        sampling_rate: float,
        n_rows: int,
        suptitle:str
):
    """
    Plot feature matrix.

    Parameters
    ----------
    input_feature:
        The (num_samples, n_neurons, num_feature) feature array.
    neuron_id:
        The neuron to plot.
    sampling_rate:
        Sampling rate in hz.
    n_rows:
        Number of rows to plot.
    suptitle:
        Suptitle of the plot.

    Returns
    -------

    """
    window_size = input_feature.shape[2]
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    time = np.arange(0, window_size) / sampling_rate
    for k in range(n_rows):
        ax = plt.subplot(n_rows, 1, k + 1)
        plt.step(time, input_feature[k, neuron_id], where="post")

        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.axvspan(0, time[-1], alpha=0.4, color="orange")
        ax.set_yticks([])
        if k != n_rows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("lag (sec)")
        if k in [0, n_rows - 1]:
            ax.set_ylabel("$t_{%d}$" % (window_size + k), rotation=0)

    plt.tight_layout()
    return fig


def plot_weighted_sum_basis(time, weights, basis_kernels, basis_coeff):
    """
    Plot weighted sum of basis.

    Parameters
    ----------
    time:
        Time axis.
    weights:
        GLM fitted weights (num_neuron, window_size).
    basis_kernels:
        Basis kernels (window_size, num_basis_funcs).
    basis_coeff:
        The basis coefficients.

    Returns
    -------
        The figure.

    """
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    axs[0].set_title("Basis")
    lines = axs[0].plot(time, basis_kernels)
    axs[0].set_xlabel("Time from spike (sec)")
    axs[0].set_ylabel("a.u.")

    colors = [p.get_color() for p in lines]

    axs[1].set_title("Coefficients")
    for k in range(len(basis_coeff)):
        axs[1].bar([k], [basis_coeff[k]], width=1, color=colors[k])
    axs[1].set_xticks([0, 7])
    axs[1].set_xlabel("Basis ID")
    axs[1].set_ylabel("Coefficient")

    axs[2].set_title("Basis x Coefficients")
    # flip time plot how a spike affects the future rate
    for k in range(basis_kernels.shape[1]):
        axs[2].plot(time, basis_kernels[:, k] * basis_coeff[k], color=colors[k])
    axs[2].set_xlabel("Time from spike (sec)")
    axs[2].set_ylabel("Weight")

    axs[3].set_title("Spike History Effect")
    axs[3].plot(time, np.squeeze(weights), alpha=0.3)
    axs[3].plot(time, basis_kernels @ basis_coeff, "--k")
    axs[3].set_xlabel("Time from spike (sec)")
    axs[3].set_ylabel("Weight")

    plt.tight_layout()
    return fig


class PlotSlidingWindow():
    def __init__(
            self,
            counts: nap.Tsd,
            n_shift: int,
            history_window: float,
            bin_size: float,
            start: float,
            ylim: tuple[float, float],
            plot_every: int,
            figsize=tuple[float,float],
            interval: int = 10,
            add_before: float = 0.2,
            add_after: float = 0.2
    ):
        self.counts = counts
        self.n_shift = n_shift
        self.history_window = history_window
        self.plot_every = plot_every
        self.bin_size = bin_size
        self.start = start
        self.ylim = ylim
        self.add_before = add_before
        self.add_after = add_after
        self.fig, self.rect_obs, self.rect_hist = self.set_up(figsize)
        self.interval = interval
        self.count_frame_0 = -1

    def set_up(self, figsize):
        fig = plt.figure(figsize=figsize)

        # set up the plot for the sliding history window
        ax = plt.subplot2grid((5, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
        # create the two rectangles, prediction and current observation
        rect_hist = plt.Rectangle((self.start, 0),  self.history_window, self.ylim[1] - self.ylim[0],
                                       alpha=0.3,
                                       color="orange")
        rect_obs = plt.Rectangle((self.start + self.history_window, 0), self.bin_size, self.ylim[1] - self.ylim[0],
                                       alpha=0.3,
                                       color="tomato")
        plot_ep = nap.IntervalSet(- self.add_before + self.start,
                                  self.start + self.history_window + self.n_shift*self.bin_size*self.plot_every +
                                  self.add_after)
        ax.step(self.counts.restrict(plot_ep).t, self.counts.restrict(plot_ep).d, where="post")
        ax.add_patch(rect_obs)
        ax.add_patch(rect_hist)
        ax.set_xlim(*plot_ep.values)

        # set up the feature matrix plot
        ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4, colspan=1, fig=fig)
        # iset = nap.IntervalSet(start=rect_hist.get_x(), end=rect_hist.get_x() + rect_hist.get_width())
        # cnt = self.counts.restrict(iset).d
        # ax.step(np.arange(cnt.shape[0]), self.n_shift * np.diff(self.ylim) + cnt, where="post")
        ax.set_ylim(0, self.n_shift * np.diff(self.ylim))
        plt.tight_layout()
        return fig, rect_obs, rect_hist

    def update_fig(self, frame):
        print(frame)
        if frame == 0:
            self.count_frame_0 += 1
        if frame == self.n_shift - 1:
            self.rect_hist.set_x(self.start)
            self.rect_obs.set_x(self.start + self.history_window)
        else:
            self.rect_obs.set_x(self.rect_obs.get_x() + self.bin_size*self.plot_every)
            self.rect_hist.set_x(self.rect_hist.get_x() + self.bin_size*self.plot_every)

            iset = nap.IntervalSet(start=self.rect_hist.get_x(), end=self.rect_hist.get_x() + self.rect_hist.get_width())
            cnt = self.counts.restrict(iset).d
            self.fig.axes[1].step(np.arange(cnt.shape[0]), np.diff(self.ylim) * (self.n_shift - frame - 1 - self.count_frame_0) + cnt, where="post")

    def run(self):
        return FuncAnimation(self.fig, self.update_fig, self.n_shift, interval=self.interval, repeat=True)
