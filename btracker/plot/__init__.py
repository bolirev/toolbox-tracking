import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import ConvexHull
import btracker.triangulate as mctri
from mpl_toolkits.mplot3d import Axes3D


def manhattan(manhattan_3d, figsize=(15, 15)):
    """Plot a manhattan dataframe

    :param manhattan_3d: x,y,z coordinates of each tower
    :param figsize: figure size
    :returns: figure and axis handles

    """
    fig = plt.figure(figsize=figsize)
    ax0 = fig.add_subplot(221)
    ax1 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    for ax, col_i, col_j in zip([ax1, ax0, ax3],
                                ['x', 'x', 'z'],
                                ['y', 'z', 'y']):
        ax.plot(manhattan_3d.loc[:, col_i], manhattan_3d.loc[:, col_j], 'ko')
        ax.set_xlabel('{} [{}]'.format(col_i, manhattan_3d.unit))
        ax.set_ylabel('{} [{}]'.format(col_j, manhattan_3d.unit))
        ax.axis('equal')

    # Annotate
    for row_i, row_val in manhattan_3d.iterrows():
        shift = [10, 10]
        xy = row_val.loc[['x', 'y']]
        xytext = xy - 2 * ((xy > 0) - 0.5) * shift
        ax1.annotate('{}'.format(row_i), xy, xytext)

    ax4 = fig.add_subplot(222, projection='3d')
    ax4.plot(manhattan_3d.loc[:, 'x'],
             manhattan_3d.loc[:, 'y'],
             manhattan_3d.loc[:, 'z'], 'ko')
    ax4.set_xlabel('{} [{}]'.format('x', manhattan_3d.unit))
    ax4.set_ylabel('{} [{}]'.format('y', manhattan_3d.unit))
    ax4.set_zlabel('{} [{}]'.format('z', manhattan_3d.unit))
    return fig, [ax0, ax1, ax3, ax4]


def error_reconstruction(npoints, edge_length, cameras_calib,
                         log10_threshold=-2,
                         figsize=(16, 8), gridsize=25):
    # Estimate reconstruction error
    error_df = mctri.error_reconstruction(npoints, edge_length, cameras_calib)
    # Esimtate good recording region
    sub_error = error_df.loc[error_df[(
        'error', 'log10_euclidian')] < log10_threshold, :]
    points = sub_error[[('reference', 'x'), ('reference',
                                             'y'), ('reference', 'z')]].values
    hull = ConvexHull(points)

    # plot
    fig = plt.figure(figsize=figsize)
    axarr = list()
    axarr.append(fig.add_subplot(242))
    axarr.append(fig.add_subplot(246))
    axarr.append(fig.add_subplot(245))
    axarr.append(fig.add_subplot(241))
    axarr.append(fig.add_subplot(243,  projection='3d'))
    axarr.append(fig.add_subplot(244))
    axarr.append(fig.add_subplot(247))
    axarr.append(fig.add_subplot(248))

    vmin = error_df[('error', 'log10_euclidian')].min()
    vmax = error_df[('error', 'log10_euclidian')].max()
    xmin = error_df[('reference', 'x')].min()
    xmax = error_df[('reference', 'x')].max()
    ymin = error_df[('reference', 'y')].min()
    ymax = error_df[('reference', 'y')].max()
    zmin = error_df[('reference', 'z')].min()
    zmax = error_df[('reference', 'z')].max()

    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ax = axarr[0]
    error_df.plot.hexbin(ax=ax,
                         x=('reference', 'x'),
                         y=('reference', 'y'),
                         C=('error', 'log10_euclidian'),
                         reduce_C_function=np.median,
                         gridsize=gridsize,
                         vmin=vmin, vmax=vmax, cmap=cmap,
                         colorbar=False)
    ax = axarr[1]
    error_df.plot.hexbin(ax=ax,
                         x=('reference', 'x'),
                         y=('reference', 'z'),
                         C=('error', 'log10_euclidian'),
                         reduce_C_function=np.median,
                         gridsize=gridsize,
                         vmin=vmin, vmax=vmax, cmap=cmap,
                         colorbar=False)
    ax = axarr[2]
    error_df.plot.hexbin(ax=ax,
                         x=('reference', 'y'),
                         y=('reference', 'z'),
                         C=('error', 'log10_euclidian'),
                         reduce_C_function=np.median,
                         gridsize=gridsize,
                         vmin=vmin, vmax=vmax, cmap=cmap,
                         colorbar=False)

    for ax in axarr[:3]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    ax = axarr[3]
    ax.semilogx(np.sort(error_df[('error', 'euclidian')]),
                100 * error_df.index / error_df.index.max())

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('error log-scale [mm]')
    ax.set_ylabel('percentage of point [%]')

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.15, 0.01, 0.3, 0.02])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                              norm=norm,
                              orientation='horizontal')
    cbar_ax.set_xlabel('error log-scale [mm]')

    ax = axarr[4]
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(points[s, 0], points[s, 1], points[s, 2], "k-")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_zlim([zmin, zmax])

    ax = axarr[5]
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(points[s, 0], points[s, 1], "k-")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ax = axarr[6]
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(points[s, 1], points[s, 2], "k-")
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_xlim([ymin, ymax])
    ax.set_ylim([zmin, zmax])

    ax = axarr[7]
    for s in hull.simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(points[s, 0], points[s, 2], "k-")
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])

    for ax in axarr[5:8]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    return fig, axarr
