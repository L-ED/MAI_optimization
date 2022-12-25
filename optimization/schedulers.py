import numpy as np

class ReduceLrOnPlateau:
    """
    ReduceLrOnPlateau scheduler scales learning rate if metric is not improved
    Scheduler controlled by:
        wait_epochs - how much epochs skip before update learning rate
        lr_scale - learning rate scale value
        threshold_percentage - 
            which percent of best value would be assumed as threshold value
    
    If absolute difference between best value and current value is less than threshold,
    then current epoch assumed as bad

    If number of bad epochs is greater than wait number, learning rate would be reduced
    """

    def __init__(self, optimizer, wait_epochs, lr_scale, threshold_percentage, mode="min"):
        
        assert mode in ['min', 'max']

        self.optimizer = optimizer
        self.learning_rate = optimizer.learning_rate
        self.wait_epochs = wait_epochs
        self.scale = lr_scale
        self.threshold = threshold_percentage
        self.bad_epochs_counter = 0
        self.mode = mode
        self.last_value = None

    
    def __call__(self, function_value):
        
        if self.last_value is None:
            self.last_value = function_value
        else:
            if self.better(function_value):
                self.last_value = function_value
                self.bad_epochs_counter = 0
            else:
                print("bad epoch")
                self.bad_epochs_counter += 1
        
        print("num bad epochs ", self.bad_epochs_counter)
        if self.bad_epochs_counter > self.wait_epochs:
            print("reducing lr")
            self.learning_rate *= self.scale
            self.optimizer.learning_rate = self.learning_rate
            self.bad_epochs_counter = 0



    def better(self, function_value):
        difference = function_value - self.last_value
        thresh_value = self.last_value*self.threshold
        if (self.mode == "min" and difference<0) or\
            (self.mode == "max" and difference>0):
                
                return abs(difference) > abs(thresh_value)
        else:
                return False


class Sinusoidal:
    """
    Sinusoidal scheduler change learning rate value according
    to Sinus function from one value

    Scheduler controlled by:
        frequency - how much epochs would be assumed as one period
    """

    def __init__(self, optimizer, frequency):

        self.optimizer = optimizer
        self.amplitude = self.optimizer.learning_rate
        self.frequency = frequency
        self.epoch_counter = 0

    
    def __call__(self):

        angle = self.epochcounter/self.frequency*np.pi
        self.optimizer.learning_rate = np.sin(angle)*self.amplitude
         

SCHEDULERS = {
    'reducelronpleteau': ReduceLrOnPlateau,
    'sinusoidal': Sinusoidal
}