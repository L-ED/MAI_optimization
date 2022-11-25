import numpy as np

# Point coordinates order is [x, y]
# Vanilla gradient descent make step after full dataset accumulation  
# (step after full derivative)
# Stochastic gradient descent split data on batches and making step after batch 
# (step after x derivative, step after y derivative) but it's not right  


# vanilla gradient descent
def GD(
        function_derivative, point, learning_rate):
    new_point = point - learning_rate*function_derivative(point)
    return  new_point

# stochastic gradient descent
def SGD(
        function_derivative, point, learning_rate):
    new_point = point[:]
    for coord in ["x", "y"]:
        new_point -= learning_rate*function_derivative(point, coord)
    return new_point

# gradient descent with previous gradient exponential accumulation
def EMA_GD(
        function_derivative, point, moment, learning_rate, gamma):
    moment = gamma*moment + learning_rate*function_derivative(point)
    new_point = point - moment
    return new_point, moment


# momentum with future point gradient lookup (Nesterov momentum)
def NAG(
        function_derivative, point, moment, learning_rate, gamma):
    future_point = point - gamma*moment
    moment = gamma*moment + learning_rate*function_derivative(
        future_point)
    new_point = point - moment
    return new_point, moment

# nesterov momentum with new moment scaling
def NAG2(
        function_derivative, point, moment, learning_rate, gamma):
    future_point = point - gamma*moment
    moment = gamma*moment +\
        learning_rate*(1-gamma)*function_derivative(future_point)
    new_point = point - moment
    return new_point, moment

# adaptive learning rate via rms(root mean squared) gradient
def AdaGrad(
        function_derivative, point, grad_l2, learning_rate):
    grad = function_derivative(point)
    grad_l2 += grad**2
    new_point = point - learning_rate*grad/(grad_l2 + 1e-8)**0.5
    return new_point, grad_l2

# adagrad with gradient vanish fix by exponential 
def RMSprop(
        function_derivative, point, grad_ema, learning_rate, gamma):
    grad = function_derivative(point)
    grad_ema = grad_ema*gamma + (1-gamma)*grad**2
    new_point = point - learning_rate*grad/(grad_ema + 1e-8)**0.5
    return new_point, grad_ema
    

OPTIMIZERS = {
    'gd': GD,
    'sgd': SGD,
    'ema_gd': EMA_GD,
    'nag': NAG,
    'nag2': NAG2,
    'adagrad': AdaGrad,
    'rmsprop': RMSprop,
}


class Optimizer:

    def __init__(
            self, function_grad, 
            method, learning_rate, 
            gamma=None):

        self.gradient = function_grad
        self.method = method
        self.optimizer = OPTIMIZERS[method]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.moment = 0

    def __call__(self, point):
        
        if self.method in ['gd', 'sgd']:
            new_point = self.optimizer(
                self.gradient,
                point,
                self.learning_rate
            )
        else:
            if self.method == "adagrad":
                new_point, self.moment = self.optimizer(
                    self.gradient,
                    point,
                    self.moment,
                    self.learning_rate
                )
            else:
                new_point, self.moment = self.optimizer(
                    self.gradient,
                    point,
                    self.moment,
                    self.learning_rate,
                    self.gamma
                )
            
        return new_point