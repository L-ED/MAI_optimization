import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_metric(metric, ax=None):
    x = range(len(metric))
    if ax is None:
        plt.plot(x, metric)
    else:
        ax.plot(x, metric)


def plot_surface(function, X, Y, ax_3d, ax_projection):
    x,y=np.meshgrid(X,Y)
    F=function(x,y)
    ax_projection.contourf(x,y,F, levels = 50)
    ax_3d.contour3D(x,y,F,500)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('F')
    ax_3d.view_init(50,50)


def plot(function, X, Y, points):

    fig = plt.figure(figsize=(9,30))
    # create 3d function surface subplot
    ax_3d = fig.add_subplot(1, 3, 1, projection='3d')
    # create surface projection on xy plane
    ax_projection = fig.add_subplot(1, 3, 2, aspect="equal")
    # create metrics plot
    ax_metric = fig.add_subplot(1, 3, 3, aspect="equal")
    ax_metric.set_title("Function value")

    metrics = points[:, 2]
    plot_surface(function, X, Y, ax_3d, ax_projection)
    ax_3d.plot(points[:, 0], points[:, 1], points[:, 2])
    plot_metric(metrics, ax_metric)
    ax_3d.scatter(points[:, 0], points[:, 1], metrics)

    ax_projection.scatter(points[:, 0], points[:, 1], metrics)



    