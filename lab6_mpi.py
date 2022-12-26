import numpy as np
from mpi4py import MPI
import argparse


def quadratic(x):
    return x**2


def linear(x):
    return x


def MonteCarloIntegration(f, low, high, discretization):
    points = np.random.uniform(low, high, discretization)
    integal = (high-low)*sum(f(points))/len(points)
    return integal


def MPI_integration(discretization, func_type, low, high):

    print("START")

    FUNCS={
        "linear":linear,
        "quadratic":quadratic
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    step_size = (high-low)/size
    low_ = low + step_size*rank
    high_ = low + step_size*(rank+1)

    integral = MonteCarloIntegration(
        FUNCS[func_type],
        low_,
        high_,
        discretization
    )

    # sum will work because we have same sample points on each process
    result = comm.reduce(integral, op=MPI.SUM, root=0)

    if rank == 0:
        result /= size

        print(f"Integral of {func_type} function on range from {low} to {high} is {result}") 


MPI_integration(1000, "linear", 0, 1)