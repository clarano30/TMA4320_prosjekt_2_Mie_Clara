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

def transition_probabilities(x0, V, beta=1.0):
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

def step(positions, V, beta=1.0):
    new_positions = positions.copy()

    for i, x0 in enumerate(positions):
        p_minus, p_0, p_plus = transition_probabilities(x0, V)

        r = np.random.rand()
        if r <= p_minus:
            new_positions[i] = x0 - h
        elif r > 1 - p_plus:
            new_positions[i] = x0 + h
        else:
            new_positions[i] = x0
    
    return new_positions

counts_sum = {}
min_x = None
max_x = None

for _ in range(N_runs):
    positions = initial_positions.copy()

    for _ in range(N_steps):
        positions = step(positions, V_constant, beta=beta_k)

    xs, counts = np.unique(positions, return_counts=True)

    if min_x is None:
        min_x, max_x = int(xs.min()), int(xs.max())
    else:
        min_x = min(min_x, int(xs.min()))
        max_x = max(max_x, int(xs.max()))

    for x, c in zip(xs, counts):
        counts_sum[int(x)] = counts_sum.get(int(x), 0) + int(c)

x_axis = np.arange(min_x, max_x + 1)
avg_counts = np.array([counts_sum.get(int(x), 0) for x in x_axis], dtype=float) / N_runs
avg_density = avg_counts / N_particles


plt.figure()
plt.plot(x_axis, avg_density, marker='o', linestyle='-')
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling (sannsynligheten)")
plt.title("Oppgave 2a(i): $V(x) = k$")
plt.grid(True, alpha=0.3)
plt.show()




###Del ii) V(x)=-kx:


def V_linear(x, k=1):
    return -k * x

counts_sum_linear = {}
min_x_linear = None
max_x_linear = None

for _ in range(N_runs):
    positions = initial_positions.copy()

    for _ in range(N_steps):
        positions = step(positions, V_linear, beta=beta_k)
    
    xs, counts = np.unique(positions, return_counts=True)

    if min_x_linear is None:
        min_x_linear, max_x_linear = int(xs.min()), int(xs.max())
    else:
        min_x_linear = min(min_x_linear, int(xs.min()))
        max_x_linear = max(max_x_linear, int(xs.max()))

    for x, c in zip(xs, counts):
        counts_sum_linear[int(x)] = counts_sum_linear.get(int(x), 0) + int(c)


x_axis_linear = np.arange(min_x_linear, max_x_linear + 1)
avg_counts_linear = np.array(
    [counts_sum_linear.get(int(x), 0) for x in x_axis_linear], dtype=float
) / N_runs
avg_density_linear = avg_counts_linear / N_particles

plt.figure()
plt.plot(x_axis_linear, avg_density_linear, marker='o', linestyle='-')
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling (sannsynlighet)")
plt.title("Oppgave 2a(ii): $V(x) = -kx$")
plt.grid(True, alpha=0.3)
plt.show()



### Exercise 2b

def step_hardcore(positions, V, beta=1.0, h=1):
    new_positions = positions.copy()
    occupied = set(new_positions.tolist())

    indices = np.random.permutation(len(new_positions))

    for i in indices:
        x0 = new_positions[i]

        p_minus, p_0, p_plus = transition_probabilities(x0, V, beta)
        r = np.random.rand()

        if r <= p_minus:
            candidate = x0 - h
        elif r > 1 - p_plus:
            candidate = x0 + h
        else:
            candidate = x0

        if candidate != x0 and candidate not in occupied:
            occupied.remove(x0)
            occupied.add(candidate)
            new_positions[i] = candidate

    return new_positions



