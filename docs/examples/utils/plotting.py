#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pynapple as nap


def current_injection_plot(current: nap.Tsd, spikes: nap.TsGroup,
                           firing_rate: nap.TsdFrame):
    ex_intervals = current.threshold(0.0).time_support


    # define plotting parameters
    # colormap, color levels and transparency level
    # for the current injection epochs
    cmap = plt.get_cmap("autumn")
    color_levs = [0.8, 0.5, 0.2]
    alpha = 0.4

    fig = plt.figure()
    # first row subplot: current
    ax = plt.subplot2grid((3, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(current, color="grey")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Injected Current")
    ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
    ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
    ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))

    # second row subplot: response
    ax = plt.subplot2grid((3, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(firing_rate, color="k")
    ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Response")
    ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
    ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
    ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))
    ylim = ax.get_ylim()

    # third subplot: zoomed responses
    for i in range(len(ex_intervals)-1):
        interval = ex_intervals.loc[[i+1]]
        ax = plt.subplot2grid((3, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
        ax.plot(firing_rate.restrict(interval), color="k")
        ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(ylim)
        for spine in ["left", "right", "top", "bottom"]:
            color = cmap(color_levs[i])
            # add transparency
            color = (*color[:-1], alpha)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)

    plt.tight_layout()
