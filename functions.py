import numpy as np
import matplotlib.pyplot as plt

def himmelblau(x, y):
    return (x**2+y-11)**2 + (x+y**2-7)**2

def himmelblau_grad(point, key=None):
    x, y = point
    grad = np.array([0, 0])
    if key is None or key=='x':
        grad[0] = 2*(x**2+y-11)*2*x + 2*(x+y**2-7)
    if key is None or key=='y':
        grad[1] = 2*(x**2+y-11) + 2*(x+y**2-7)*2*y
    return grad


def eggholder(x, y):
    return -(y-47)*np.sin((np.abs(x/2 + y + 47))**0.5) - x*np.sin((np.abs(x/2 - y - 47))**0.5)

def eggholder_grad(point, key=None):

    x, y = point
    grad = np.array([0, 0])
    subgrad_pt1 = np.cos((np.abs(x/2 + y + 47))**0.5)*0.5*((np.abs(x/2 + y + 47))**(-0.5))*np.sign(x/2 + y + 47)
    subgrad_pt2 = np.cos((np.abs(x/2 - y - 47))**0.5)*0.5*((np.abs(x/2 - y - 47))**(-0.5))*np.sign(x/2 - y - 47)
    if key is None or key=='x':
        grad[0] = -(y-47)*subgrad_pt1*0.5 - (np.sin((np.abs(x/2 - y - 47))**0.5) + subgrad_pt2*0.5)
    if key is None or key=='y':
        grad[1] = -(np.sin((np.abs(x/2 + y + 47))**0.5) + (y-47)*subgrad_pt1) +  subgrad_pt2
    return grad


FUNCTIONS = {
    'himmelblau':
        {
            "function":himmelblau,
            "gradient":himmelblau_grad,
            "X": np.linspace(-5, 5),
            "Y": np.linspace(-5, 5)
        },
    'eggholder':
        {
            "function":eggholder,
            "gradient":eggholder_grad,
            "X": np.linspace(-500, 500, 200),
            "Y": np.linspace(-500, 500, 200)
        }

}