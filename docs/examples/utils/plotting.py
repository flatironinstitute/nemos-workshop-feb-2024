#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import jax
import pynapple as nap
import pandas as pd
import matplotlib as mpl
from typing import Optional


def tuning_curve_plot(tuning_curve: pd.DataFrame):
    fig, ax = plt.subplots(1, 1)
    tc_idx = tuning_curve.index.to_numpy()
    tc_val = tuning_curve.values.flatten()
    width = tc_idx[1]-tc_idx[0]
    ax.bar(tc_idx, tc_val, width, facecolor="grey", edgecolor="k",
           label="observed", alpha=0.4)
    ax.set_xlabel("Current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    return fig


def current_injection_plot(current: nap.Tsd, spikes: nap.TsGroup,
                           firing_rate: nap.TsdFrame,
                           predicted_firing_rate: Optional[nap.TsdFrame] = None):
    ex_intervals = current.threshold(0.0).time_support

    # define plotting parameters
    # colormap, color levels and transparency level
    # for the current injection epochs
    cmap = plt.get_cmap("autumn")
    color_levs = [0.8, 0.5, 0.2]
    alpha = 0.4

    fig = plt.figure(figsize=(7, 7))
    # first row subplot: current
    ax = plt.subplot2grid((4, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(current, color="grey")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Injected Current")
    ax.set_xticklabels([])
    ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
    ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
    ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))

    # second row subplot: response
    resp_ax = plt.subplot2grid((4, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
    resp_ax.plot(firing_rate, color="k", label="Observed firing rate")
    if predicted_firing_rate:
        resp_ax.plot(predicted_firing_rate, color="tomato", label='Predicted firing rate')
    resp_ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10, label="Observed spikes")
    resp_ax.set_ylabel("Firing rate (Hz)")
    resp_ax.set_xlabel("Time (s)")
    resp_ax.set_title("Neural response", y=.95)
    resp_ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
    resp_ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
    resp_ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))
    ylim = resp_ax.get_ylim()

    # third subplot: zoomed responses
    zoom_axes = []
    for i in range(len(ex_intervals)-1):
        interval = ex_intervals.loc[[i+1]]
        ax = plt.subplot2grid((4, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
        ax.plot(firing_rate.restrict(interval), color="k")
        ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
        if predicted_firing_rate:
            ax.plot(predicted_firing_rate.restrict(interval), color="tomato")
        else:
                ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        for spine in ["left", "right", "top", "bottom"]:
            color = cmap(color_levs[i])
            # add transparency
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)
        zoom_axes.append(ax)

    resp_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.4),
                   bbox_transform=zoom_axes[1].transAxes)


def lnp_schematic(input_feature: nap.Tsd,
                  weights: np.ndarray,
                  intercepts: np.ndarray,
                  plot_nonlinear: bool = False,
                  plot_spikes: bool = False):
    """Create LNP schematic.

    - Works best with len(weights)==3.

    - Requires len(weights)==len(intercepts)

    - plot_nonlinear=False and plot_spikes=True will look weird

    """
    assert len(weights) == len(intercepts), "weights and intercepts must have same length!"
    fig, axes = plt.subplots(len(weights), 4, sharex=True,
                             sharey='col',
                             gridspec_kw={'wspace': .65})
    for ax in axes.flatten():
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    weights = np.expand_dims(weights, -1)
    intercepts = np.expand_dims(intercepts, -1)
    times = input_feature.t
    # only need to do this once, since they share x
    axes[0, 0].set_xticks([times.min(), times.max()])
    input_feature = np.expand_dims(input_feature, 0)
    linear = weights * input_feature + intercepts
    axes[0, 0].set_visible(False)
    axes[2, 0].set_visible(False)
    axes[1, 0].plot(times, input_feature[0], 'gray')
    axes[1, 0].set_title('$x$', fontsize=10)
    axes[1, 0].tick_params('x', labelbottom=True)
    axes[0, 1].tick_params('y', labelleft=True)
    axes[2, 1].tick_params('y', labelleft=True)
    arrowkwargs = {'xycoords': 'axes fraction', 'textcoords': 'axes fraction',
                   'ha': 'center', 'va': 'center'}
    arrowprops = {'color': '0', 'arrowstyle': '->', 'lw': 1,
                  'connectionstyle': 'arc,angleA=0,angleB=180,armA=20,armB=25,rad=5'}
    y_vals = [1.7, .5, -.7]
    for y in y_vals:
        axes[1, 0].annotate('', (1.5, y), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    titles = []
    for i, l in enumerate(linear):
        axes[i, 1].plot(times, l)
        if intercepts[i, 0] < 0:
            s = '-'
        else:
            s = '+'
        titles.append(f"{weights[i, 0]}x {s} {abs(intercepts[i, 0])}")
        axes[i, 1].set_title(f"${titles[-1]}$", y=.95, fontsize=10)
    nonlinear = np.exp(linear)
    if plot_nonlinear:
        for i, l in enumerate(nonlinear):
            axes[i, 2].plot(times, l)
            axes[i, 2].set_title(f"$\\exp({titles[i]})$", y=.95, fontsize=10)
            axes[i, 1].annotate('', (1.5, .5), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    else:
        for i, _ in enumerate(nonlinear):
            axes[i, 2].set_visible(False)
    if plot_spikes:
        for i, l in enumerate(nonlinear):
            gs = axes[i, 3].get_subplotspec().subgridspec(3, 1)
            axes[i, 3].set_frame_on(False)
            axes[i, 3].xaxis.set_visible(False)
            axes[i, 3].yaxis.set_visible(False)
            for j in range(3):
                ax = fig.add_subplot(gs[j, 0])
                spikes = jax.random.poisson(jax.random.PRNGKey(j*i + j + i), l)
                spike_times = np.where(spikes)
                spike_heights = spikes[spike_times]
                ax.vlines(times[spike_times], 0, spike_heights, color='k')
                ax.yaxis.set_visible(False)
                if j != 2 or i != len(nonlinear)-1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.set_xticks([times.min(), times.max()])
            axes[i, 2].annotate('', (1.5, .5), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    else:
        for i, _ in enumerate(nonlinear):
            axes[i, 3].set_visible(False)
    suptitles = ["Input", "Linear", "Nonlinear", "Poisson samples\n(spikes)"]
    suptitles_to_add = [True, True, plot_nonlinear, plot_spikes]
    for b, ax, t in zip(suptitles_to_add, axes[0, :], suptitles):
        if b:
            axes[0,1].text(.5, 1.4, t, transform=ax.transAxes,
                           horizontalalignment='center',
                           verticalalignment='top', fontsize=12)
    return fig
