import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys 
import argparse

sys.path.insert(0, "/home/led/MAI/")
from optimization.optimizers import *
from optimization.schedulers import *
from optimization.optimization_loops import optimize
from optimization.functions import *
from optimization.utils import *



def names_to_numbered_string(names):
    final_string = ''
    for i, name in enumerate(names):
        num = i#+1
        name_str = f"\t{num}. {name}"
        if num%3==0:
            name_str += "\n"
        
        final_string +=name_str
    return final_string


if __name__ == "__main__":

    functions_names = list(FUNCTIONS.keys())
    optimizers_names = list(OPTIMIZERS.keys())
    schedulers_names = list(SCHEDULERS.keys())

    parcer = argparse.ArgumentParser(
        prog="Function optimization",
        description="Optimizing chosen function by name with chosen optimizer and scheduler"
    )

    parcer.add_argument(
        "-f", "--function", type=int,
        help="function name index for optimization\n"+\
            names_to_numbered_string(functions_names)
    )

    parcer.add_argument(
        "-o", "--optimizer", type=int, 
        help="Optimizer name index\n"+\
            names_to_numbered_string(optimizers_names)
    )

    parcer.add_argument(
        "-sp", "--start_point", type=float, nargs=2, action="extend",
        help="Coordinate of starting point"
    )

    parcer.add_argument(
        "-e", "--epochs", type=int, default=50,
        help="How much steps optimizer do"
    )

    parcer.add_argument(
        "-g", "--gamma", type=float, default=0.9,
        help="Gamma parameter for momentum optimizers"
    )

    parcer.add_argument(
        "-lr", "--learning_rate", type=float, default=0.01,
        help="Learning rate regulate optimizer coodinate step length"
    )

    parcer.add_argument(
        "-s", "--scheduler", type=int, default=0,
        help="Scheduler name index\n"+\
            names_to_numbered_string(schedulers_names)
    )

    parcer.add_argument(
        "-we", "--wait_epochs", type=int, default=10,
        help="Scheduler parameter that regulate:"+\
            "\t number of warmup epochs after last learning rate scaling for ReduceLrOnPlateau"+\
            "\t Frequency parameter of Sinusoidal scheduler (how much epochs will be a full cycle)"
    )

    parcer.add_argument(
        "-s_lr", "--scale_lr", type=float, default=0.9,
        help="Learning rate scale parameter for ReduceLrOnPlateau scheduler"
    )

    parcer.add_argument(
        "-thr", "--threshold", type=float, default=0.005,
        help="Threshold value relative to previous best metric value (thresh_value = best_value*threshold)."+\
            "if difference between best metric and current less than threshold, than current epoch is bad.s"+\
            "Only for ReduceLrOnPlateau scheduler"
    )

    args = parcer.parse_args()

    func_dict = FUNCTIONS[functions_names[args.function]]
    function = func_dict["function"]
    X = func_dict["X"]
    Y = func_dict["Y"]
    gradient = func_dict["gradient"]

    optimizer = Optimizer(
        function_grad=gradient,
        method=optimizers_names[args.optimizer],
        learning_rate=args.learning_rate,
        gamma=args.gamma
    )

    # scheduler = SCHEDULERS[schedulers_names[args.scheduler]]
    
    if args.scheduler ==0:
        # print(scheduler)
        scheduler = SCHEDULERS[
            schedulers_names[args.scheduler]](
                optimizer=optimizer,
                wait_epochs = args.wait_epochs, 
                lr_scale = args.scale_lr, 
                threshold_percentage = args.threshold
        )

    else:
        scheduler = SCHEDULERS[
            schedulers_names[args.scheduler]](
            optimizer=optimizer,
            frequency=args.wait_epochs,
        )



    fig = plt.figure(figsize=(12,6))
    fig.suptitle("Function view")
    ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_projection = fig.add_subplot(1, 2, 2, aspect="equal")
    plot_surface(function, X, Y, ax_3d, ax_projection)

    plt.show(block=False)
    plt.pause(0.1)


    coords = input("Input starting point as X,Y\n")
    coords = np.array([float(i) for i in coords.split(",")])

    plt.close()

    print(args.start_point)

    try:
        points = optimize(
            init_point= np.array(coords),#args.start_point,
            epochs= args.epochs,
            function=function,
            optimizer=optimizer,
            scheduler=scheduler
        )

        plot(
            function=function,
            X=X, Y=Y, points=points
        )

    except OverflowError:
        print('Gradient out of bounds, try to reduce learning rate')

    

    plt.show()



    


    