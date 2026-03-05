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
        p_minus, p_0, p_plus = transition_probabilities(x0, V, beta)
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


def run_simulation_avg_density(step_fn, V_fn, beta=1.0):
    counts_sum = {}
    min_x = None
    max_x = None

    for _ in range(N_runs):
        positions = initial_positions.copy()

        for _ in range(N_steps):
            positions = step_fn(positions, V_fn, beta=beta, h=h)

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

    return x_axis, avg_density


# Graf for 2b(i) og for V(x) = k 
x_axis_b1, avg_density_b1 = run_simulation_avg_density(
    step_fn=step_hardcore,
    V_fn=V_constant,
    beta=beta_k
)

plt.figure()
plt.plot(x_axis_b1, avg_density_b1, marker='o', linestyle='-')
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling (sannsynlighet)")
plt.title("Oppgave 2b(i): Hard-core, $V(x)=k$")
plt.grid(True, alpha=0.3)
plt.show()


# Graf for 2b(ii)for V(x) = -kx

x_axis_b2, avg_density_b2 = run_simulation_avg_density(
    step_fn=step_hardcore,
    V_fn=V_linear,
    beta=beta_k
)

plt.figure()
plt.plot(x_axis_b2, avg_density_b2, marker='o', linestyle='-')
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling (sannsynlighet)")
plt.title("Oppgave 2b(ii): Hard-core, $V(x)=-kx$")
plt.grid(True, alpha=0.3)
plt.show()

# Her sammenligner vi 2a og 2b sine grafer
# V(x) = k
plt.figure()
plt.plot(x_axis, avg_density, marker='o', linestyle='-', label="2a: No interaction")
plt.plot(x_axis_b1, avg_density_b1, marker='o', linestyle='--', label="2b: Hard-core")
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling")
plt.title("Comparison: V(x) = k")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# V(x) = -kx
plt.figure()
plt.plot(x_axis_linear, avg_density_linear, marker='o', linestyle='-', label="2a: No interaction")
plt.plot(x_axis_b2, avg_density_b2, marker='o', linestyle='--', label="2b: Hard-core")
plt.xlabel("x")
plt.ylabel("Gjennomsnittlig partikkelfordeling")
plt.title("Comparison: V(x) = -kx")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

#Oppgave 3a

Nx = 100
L = 2 * Nx
Tp = 500
T_total = 20 * Tp
k = 1.0
beta_k = 1000
beta = beta_k / k
Np = 12 * Nx
rng = np.random.default_rng(0)



def V1_sawtooth(x, Nx=100, alpha=0.8, k=1):
    x = np.asarray(x)

    u = np.mod(x, Nx).astype(float)

    y = np.where(u <= alpha * Nx, u, u - Nx)

    V = np.zeros_like(x, dtype=float)

    mask_pos = y > 0
    V[mask_pos] = k * (y[mask_pos] / (alpha * Nx)) #fra ligning 9 i oppgaven.

    mask_neg = ~mask_pos
    V[mask_neg] = (-k) * (y[mask_neg] / ((1 - alpha) * Nx)) #igjen fra ligning 9.

    return V

def step_flat(pos):
    r = rng.random(size=pos.size)
    move = np.zeros_like(pos, dtype=int)
    move[r < 1/3] = -1
    move[r > 2/3] = +1
    n_minus = np.count_nonzero(move == -1)
    n_plus = np.count_nonzero(move == +1)
    pos = (pos + move) % L
    return pos, n_plus, n_minus

def step_saw(pos, alpha):
    xminus = (pos - 1) % L
    x0 = pos
    xplus = (pos + 1) % L

    Vminus = V1_sawtooth(xminus, Nx, alpha = alpha, k = k)
    V0 = V1_sawtooth(x0, Nx, alpha = alpha, k = k)
    Vplus = V1_sawtooth(xplus, Nx, alpha = alpha, k = k)

    Vmin = np.minimum(np.minimum(Vminus, V0), Vplus) #Prøver å unngå overflow ved å trekke fra min før eksponentiering. 

    wminus = np.exp(-beta * (Vminus - Vmin))
    w0 = np.exp(-beta * (V0 - Vmin))
    wplus = np.exp(-beta * (Vplus - Vmin))

    Z = wminus + w0 + wplus

    pminus = wminus / Z
    pplus = wplus / Z

    r = rng.random(size=pos.size)
    move = np.zeros_like(pos, dtype=int)
    move[r <= pminus] = -1
    move[r > (1 - pplus)] = +1

    nminus = np.count_nonzero(move == -1)
    nplus = np.count_nonzero(move == +1)
    pos = (pos + move) % L
    return pos, nplus, nminus


def simulate_a(alpha):
    pos = np.repeat(np.arange(L), Np // L)

    J_t = np.zeros(T_total, dtype=float)

    for t in range(T_total):
        in_V2 = ((t // Tp) % 2 == 0) #starter i V2 med t=0, og vi bytter for hver Tp.
        if in_V2:
            pos, nplus, nminus = step_flat(pos)
        else:
            pos, nplus, nminus = step_saw(pos, alpha)
        
        J_t[t] = (nplus - nminus) / Np

    J_avg = np.array([J_t[n*(2*Tp):(n+1)*(2*Tp)].mean() for n in range(10)])
    return J_t, J_avg


results = {}

for alpha in [0.8, 0.1]:
    J_t, J_avg = simulate_a(alpha)
    results[alpha] = (J_t, J_avg)

    print(f"\nalpha = {alpha}:")
    for i, val in enumerate(J_avg):
        print(f"  J_avg({i}) = {val:.6e}")

for alpha in [0.8, 0.1]:
    J_t, J_avg = results[alpha]


#oppgave 3b

alpha = 0.8
Np = 40 * Nx #for 3b
L = 2 * Nx

Tp_values = np.linspace(1, 1001, 50, dtype=int) #Fordi vi trenger 50 verdier mellom 1 og 1001 som er jevnt fordelt. 

def init_two_minima(Np, Nx, L):
    pos = np.empty(Np, dtype=int)
    pos[:Np//2] = 0
    pos[Np//2:] = Nx
    return pos % L

def sim_one_cyc_for_Tp(Tp, alpha):
    pos = init_two_minima(Np, Nx, L)

    T_cycle = 2 * Tp
    J_t = np.zeros(T_cycle, dtype=float)

    for t in range(T_cycle):
        in_V2 = ((t//Tp) % 2 == 0) #her starter vi i V2, og bytter for hver Tp.
        if in_V2:
            pos, nplus, nminus = step_flat(pos)
        else:
            pos, nplus, nminus = step_saw(pos, alpha)

        J_t[t] = (nplus - nminus) / Np

    J_avg = J_t.mean() #som gir cycle-averaged current over en hel syklus.
    return J_avg

Javg_values = np.array([sim_one_cyc_for_Tp(Tp, alpha) for Tp in Tp_values])


plt.figure()
plt.plot(Tp_values, Javg_values, marker='o')
plt.xlabel("Tp")
plt.ylabel("J_avg (en syklus)")
plt.title("Oppgave 3b: Cycle-averaged current vs Tp for alpha=0.8")
plt.grid(True, alpha=0.3)
plt.show()


#oppgave 3c

from scipy.special import erfc

Tp = 500
Np = 12* Nx #for 3c
alpha_values = np.linspace(0, 1, 52)[1:-1] #np.linspace(0, 1, 50) fikk problemer med runtime, så bytter den til den nye. 

def Javg_analytical(alpha, Tp, Nx):
    a1 = (alpha * Nx) / (2 * np.sqrt(Tp / 3))
    a2 = ((1- alpha) * Nx) / (2 * np.sqrt(Tp / 3))
    return (Nx / (4 *Tp)) * (erfc(a1) - erfc(a2))

def sim_one_cyc_for_alpha(alpha):
    pos = np.repeat(np.arange(L), Np // L) #6 partikler for hver punkt når Np=12Nx og L=2Nx

    T_cycle = 2 * Tp
    J_t = np.zeros(T_cycle, dtype=float)

    for t in range(T_cycle):
        in_V2 = ((t // Tp) % 2 == 0)
        if in_V2:
            pos, nplus, nminus = step_flat(pos)
        else:
            pos, nplus, nminus = step_saw(pos, alpha)
        J_t[t] = (nplus - nminus) / Np

    return J_t.mean()

J_num = np.array([sim_one_cyc_for_alpha(alpha) for alpha in alpha_values]) 
J_ana = np.array([Javg_analytical(alpha, Tp, Nx) for alpha in alpha_values])

plt.figure()
plt.plot(alpha_values, J_num, marker='o', label='Numerisk (1 Syklus)')
plt.plot(alpha_values, J_ana, marker='x', label='Analytisk (Eq 15)')
plt.xlabel("alpha")
plt.ylabel("J_avg")
plt.title("Oppgave 3c: J_avg vs alpha for Tp=500")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()





# Oppgave 3d

#har samme verdier for Tp, Np, og alpha values som i 3c.

beta_k_values = [0.01, 1, 2, 3, 5, 10]

plt.figure(figsize=(8,5))

linestyles = ['-', '--', '-.', ':', '-', '--'] 
#gjør dette siden første gang jeg fikk figure, 
#så var det vanskelig å se forskjellen mellom de forskjellige kurvene.

for i, beta_k in enumerate(beta_k_values):
    beta = beta_k / k
    J_num = np.array([sim_one_cyc_for_alpha(alpha) for alpha in alpha_values])

    plt.plot(alpha_values, J_num,
             linestyle=linestyles[i],
             linewidth=2,
             label=f'Numerisk beta k={beta_k}')

#analytiske kurver (samme som i 3c):

plt.plot(alpha_values, J_ana,
         color='black',
         linewidth=3,
         label='Analytisk (Eq 15)')

plt.xlabel("alpha")
plt.ylabel("J_avg")
plt.title("Oppgave 3d: J_avg vs alpha for ulike beta_k")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout() #prøver bare å gjøre layouten litt bedre, siden det var litt mange kurver. 
plt.show()



# oppgave 3e

Nx_old, L_old, Np_old = Nx, L, Np #lagrer de gamle verdiene for å kunne bruke dem senere.

Nx = 10
L= 2 * Nx
alpha = 0.8
Np = 40 * Nx
Tp_values = np.linspace(80, 1500, 20, dtype=int) #oppgaven spør om å bruke 20 verdier mellom 80 og 1500.


def sim_one_cyc_for_Tp_3e(Tp):
    pos = init_two_minima(Np, Nx, L)
    J_t = np.zeros(2*Tp, dtype=float)
    for t in range(2*Tp):
        in_V2 = ((t//Tp) % 2 == 0)
        if in_V2:
            pos, nplus, nminus = step_flat(pos)
        else:
            pos, nplus, nminus = step_saw(pos, alpha)

        J_t[t] = (nplus - nminus) / Np

    return J_t.mean()

J_num_3e = np.array([sim_one_cyc_for_Tp_3e(Tp) for Tp in Tp_values])

J_ana_3e = np.array([Javg_analytical(alpha, Tp, Nx) for Tp in Tp_values])

plt.figure()
plt.plot(Tp_values, J_num_3e, marker='o', label='Numerisk (1 Syklus)')
plt.plot(Tp_values, J_ana_3e, marker='x', label='Analytisk (Eq 15)')
plt.xlabel("Tp")
plt.ylabel("J_avg")
plt.title("Oppgave 3e: J_avg vs Tp for alpha=0.8, Nx=10")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


Nx, L, Np = Nx_old, L_old, Np_old #setter tilbake de gamle verdiene for å kunne bruke dem i andre deler av oppgaven.