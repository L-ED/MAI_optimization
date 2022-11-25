import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import sys 
sys.path.insert(0, "/home/led/MAI/")
from optimization import *

def plot(function, X, Y):
    x,y=np.meshgrid(X,Y)
    F=function(x,y)

    fig = plt.figure(figsize=(9,4))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # ax=plt.axes(projection='3d')

    ax2 = fig.add_subplot(1, 2, 2, aspect='equal')
    # colors = matplotlib.cm.jet(np.hypot(x,y))
    # ax2.contourf(x,y,F, facecolors=colors)
    ax2.contourf(x,y,F, levels = 50)

    ax.contour3D(x,y,F,500)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('F')
    ax.view_init(50,50)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Dunction plotter")
    parser.add_argument("-f", "--function")

    args = parser.parse_args()

    if args.function == "himmelblau":
        X = np.linspace(-5, 5)
        Y = np.linspace(-5, 5)
        plot(himmelblau, X, Y)

    elif args.function == "eggholder":
        X = np.linspace(-500, 500, 200)
        Y = np.linspace(-500, 500, 200)
        plot(eggholder, X, Y)