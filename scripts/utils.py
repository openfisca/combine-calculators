import matplotlib.pyplot as plt
import numpy as np


def show_histogram(variable, legend=''):
    hist, bins = np.histogram(variable, bins=20)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center,
            hist,
            align='center',
            width=width,
            label=legend)
    if legend != '':
        plt.legend(loc='upper right')
    plt.show()


def percent_diff(a, b):
    return (a - b) / (max(a + 1, b + 1))


def scatter_plot(x, y, legx, legy, alpha=1):
    fig = plt.figure()
    ax1 = fig.add_axes((0.5,0.5,1,1))
    ax1.set_xlabel(legx)
    ax1.set_ylabel(legy)
    ax1.scatter(
        x,
        y,
        alpha=alpha)


def multi_scatter(title, xlabel, ylabel, plots):
    fig = plt.figure()
    ax = fig.add_axes((0.5,0.5,1,1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for plot in plots:
        ax.scatter(
            plot['x'],
            plot['y'],
            alpha=plot.get('alpha', 1),
            label=plot['label'],
            c=[plot['color']] * len(plot['x']))
    ax.legend(loc='upper right')
    plt.title(title)
