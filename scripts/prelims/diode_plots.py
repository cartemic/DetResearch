# -*- coding: utf-8 -*-
"""
PURPOSE:
    Generates plots of diode data for use in my preliminary proposal

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

from funcs import diodes
from matplotlib import pyplot as plt
import palettable
import numpy as np
import os


def figure(
        xlabel,
        ylabel,
        plot_title,
        axis_color
):
    # formatting
    axis_font_size = 10
    line_thk = 1.5

    # set colors
    bmap = palettable.tableau.get_map('ColorBlind_10')

    # format text
    plt.rc('text', color=axis_color)
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)

    # define figure and axes
    fig = plt.figure(figsize=(7.5, 2.5))
    axes = fig.add_subplot(111)

    # format axes
    axes.tick_params(labelsize=axis_font_size)
    axes.set_prop_cycle('color', bmap.mpl_colors)

    # set labels
    plt.xlabel(xlabel,
               fontsize=axis_font_size * 1.125,
               fontweight='bold',
               color=axis_color)
    plt.ylabel(ylabel,
               fontsize=axis_font_size * 1.125,
               fontweight='bold',
               color=axis_color)
    plt.title(plot_title,
              fontsize=axis_font_size * 1.5,
              fontweight='bold',
              color='k')

    # format axis lines
    for ctr, spine in enumerate(axes.spines.values()):
        spine.set_color(axis_color)
        if ctr % 2:
            spine.set_visible(False)
        else:
            spine.set_linewidth(line_thk)

    plt.tight_layout()

    return [fig, axes]


if __name__ == "__main__":

    # set axis color
    ax_color = '#333333'

    # locate data
    path = os.path.join(
        'C:\\',
        '20180912',
        'AR'
    )
    data_location = diodes.find_diode_data(path)[2]
    data = diodes.load_diode_data(
        data_location,
        apply_lowpass=True
    )

    # build time array and differentiate data
    time = np.array(list(range(len(data)))) * 1e-6
    diffs = data.diff(axis=0)

    x_lab = '\\textbf{Time (s)}'
    y_lab = '\\textbf{Diode Signal (V)}'
    title = '\\textbf{Wave Speed Measurement}'

    [plot, axis] = figure(
        x_lab,
        y_lab,
        title,
        ax_color
    )

    axis.plot(time, data)
    axis.legend(['Diode 0', 'Diode 1'],
                loc=4,
                edgecolor='none')
    line_times = diffs.idxmax(axis=0).values * 1e-6
    [axis.axvline(t, color=ax_color) for t in line_times]
    axis.set_xlim([0.513, 0.51375])
    axis.set_ylim([0, 0.5])

    gap_width = np.diff(line_times)[0]

    # draw arrows
    arrow_kwargs = {
        'color': ax_color,
        'length_includes_head': True,
        'head_width': 0.01,
        'head_length': gap_width / 10.,
        'overhang': 0.5
    }
    axis.arrow(line_times[0] + gap_width / 2,
               0.25,
               gap_width / 2 * 0.9,
               0,
               **arrow_kwargs)
    axis.arrow(line_times[0] + gap_width / 2,
               0.25,
               -gap_width / 2 * 0.9,
               0,
               **arrow_kwargs)
    axis.text(line_times[0] + gap_width / 2,
              0.25,
              '$\Delta t$',
              bbox={'facecolor': 'white',
                    'edgecolor': 'none'},
              horizontalalignment='center',
              verticalalignment='center',
              color=ax_color
              )

    # save plot
    plot.tight_layout()
    plot.savefig('diode_dt.svg')  # , format='pdf')

    # ------------------------------

    x_lab = '\\textbf{Time (s)}'
    y_lab = '\\textbf{Relative}\n' \
            '\\textbf{Stepwise} ' \
            '$\\boldmath{\\Delta V}$\n' \
            '\\textbf{(Single Diode)}'
    title = '\\textbf{Locating Detonations by Maximum Derivative}'

    [plot, axis] = figure(
        x_lab,
        y_lab,
        title,
        ax_color
    )

    norm_diffs = diffs[diffs.columns[0]] / diffs.max()[diffs.columns[0]]
    norm_diffs.index = norm_diffs.index * 1e-6
    min_time = 0.45
    max_time = 0.75

    axis.plot(norm_diffs[
                  (norm_diffs.index > min_time) &
                  (norm_diffs.index <= max_time)
              ])
    axis.set_xlim([min_time,
                   max_time])
    axis.set_ylim([-0.25, 1])

    # save plot
    plot.tight_layout()
    plot.savefig('max_derivative.svg')  # , format='pdf')

    # --------------------------------------

    x_lab = '\\textbf{Time (s)}'
    y_lab = '\\textbf{Diode Signals (V)}'
    title = '\\textbf{Raw Diode Trace}'

    [plot, axis] = figure(
        x_lab,
        y_lab,
        title,
        ax_color
    )

    axis.plot(time, data)
    axis.legend(['Diode 0', 'Diode 1'],
                loc='upper left',
                bbox_to_anchor=(0, 1.1),
                facecolor='none',
                edgecolor='none')
    axis.set_xlim([0.45, 0.75])
    axis.set_ylim([-1, 0.6])

    # save plot
    plot.tight_layout()
    plot.savefig('raw_diode_signal.svg')  # , format='pdf')
