import matplotlib.pyplot as plt
import numpy as np

"""
Optimize state from initial point during epochs using 
    function for metric plot
    function derivative (grad) for step direction 
    optimizer for making steps
    scheduler for learning rate tuning

"""
def optimize(init_point, epochs, function, optimizer, scheduler=None):

    points = np.array([[*init_point, function(*init_point)]])
    position = init_point
    for epoch in range(epochs):
        print("Epoch ", epoch)
        position = optimizer(position)
        metric = function(*position)
        if scheduler is not None:
            scheduler(metric)

        new_point = [[*position, metric]]
        points = np.append(points, new_point, axis=0)

    return points


