"""plotting utility functions"""

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba as trgba
from .analysis import qnorm

palette = [trgba('blue'), trgba('orange'), trgba('red'), trgba('green'), trgba('purple')]

def color_plots(x, y, c, cmap = cm.viridis, **kwargs):
    """
    Accept x,y values array and array, plot colored by c array
    args:
        x : np.array of x axis values (N,)
        y : np.array of y axis values (M, N)
        c : np.array of values on which to base coloring, numerical
        cmap : callable that accepts sequence of normalized numbers and returns color values
        kwargs : arguments for plt.plot
    returns:
        None
    """
    c = cmap(qnorm(c))
    for xx, yy, cc in zip(x, y, c):
        plt.plot(xx, yy, color = cc, **kwargs)
    return

