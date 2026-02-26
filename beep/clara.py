import numpy as np
import matplotlib.pyplot as plt


##Exercise 2

###Exercise 2a)

N_particles = 101
N_steps = 200
h = 1
beta_k = 1
N_runs = 100

initial_positions = np.arange(N_particles)

###skal starte med i) V(x)=k:

def V_constant(x,k=1):
    return k

def transition_probabilities(x0, V, beta=1):
    V_minus = V(x0 - 1)
    V_0 = V(x0)
    V_plus = V(x0 + 1)

    w_minus = np.exp(-beta * V_minus)
    w_0 = np.exp(-beta * V_0)
    w_plus = np.exp(-beta * V_plus)

    Z = w_minus + w_0 + w_plus

    p_minus = w_minus / Z
    p_0 = w_0 / Z
    p_plus = w_plus / Z

    return p_minus, p_0, p_plus

def step(positions, V):
    new_positions = positions.copy()

    for i, x0 in enumerate(positions):
        p_minus, p_0, p_plus = transition_probabilities(x0, V)

        r = np.random.rand()
        if r <= p_minus:
            new_positions[i] = x0 - 1
        elif r > 1 - p_plus:
            new_positions[i] = x0 + 1
        else:
            new_positions[i] = x0
    
    return new_positions

positions = initial_positions.copy()
for _ in range(N_steps):
    positions = step(positions, V_constant)

