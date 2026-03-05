### oppgave 4a



import numpy as np
import matplotlib.pyplot as plt


beta = 1000.0          
Nx = 20
alpha = 0.2
Tp = 40

n_peaks = 4
L = Nx * n_peaks       # 80 totalt
n_cycles = 5
T = n_cycles * Tp      # 200 steg

# Valgte verdier 
N = 10                 
b = 2                  

def sawtooth_V_at(x, Nx=20, alpha=0.2, k=1.0):
    """
    Sawtooth potential over one period Nx, extended periodically.
    Returns V(x) for integer x (site index).
    """
    xm = x % Nx
    m = int(round(alpha * Nx))
    m = max(1, min(Nx - 1, m))

    
    if xm < m:
        return k * (xm / m)                         
    else:
        return k * ((Nx - xm) / (Nx - m))          


def flashing_V_at(x, t, Tp=40, Nx=20, alpha=0.2, k=1.0):
    """
    Flashing: ON for half period, OFF for half period.
    """
    on = (t % Tp) < (Tp // 2)
    if on:
        return sawtooth_V_at(x, Nx=Nx, alpha=alpha, k=k)
    else:
        return 0.0



def ring_distance(a, c, L):
    d = abs(a - c)
    return min(d, L - d)


def allowed_position(candidate, positions, i, b, L):
    """
    True if particle i can be at 'candidate' without being within
    distance <= b of any other particle, on a ring of length L.
    """
    for j, xj in enumerate(positions):
        if j == i:
            continue
        if ring_distance(candidate, xj, L) <= b:
            return False
    return True


def transition_probabilities(x0, t, beta, L, Tp, Nx, alpha, k=1.0):
    xL = (x0 - 1) % L
    xR = (x0 + 1) % L

    V0 = flashing_V_at(x0, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)
    VL = flashing_V_at(xL, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)
    VR = flashing_V_at(xR, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)

    wL = np.exp(-beta * (VL - V0))
    wR = np.exp(-beta * (VR - V0))
    w0 = 1.0

    Z = wL + w0 + wR
    p_minus = wL / Z
    p_plus = wR / Z
    p_0 = w0 / Z
    return p_minus, p_0, p_plus


def step_hardcore_sizeb(positions, t, beta, b, L, Tp, Nx, alpha, k=1.0):
    positions = positions.copy()
    order = np.random.permutation(len(positions))

    for i in order:
        x0 = positions[i]
        p_minus, p_0, p_plus = transition_probabilities(
            x0, t, beta=beta, L=L, Tp=Tp, Nx=Nx, alpha=alpha, k=k
        )

        r = np.random.rand()
        if r <= p_minus:
            cand = (x0 - 1) % L
        elif r > 1.0 - p_plus:
            cand = (x0 + 1) % L
        else:
            cand = x0

        if cand != x0 and allowed_position(cand, positions, i, b=b, L=L):
            positions[i] = cand

    return positions


def simulate(positions0, T, beta, b, L, Tp, Nx, alpha, k=1.0):
    traj = np.zeros((T + 1, len(positions0)), dtype=int)
    traj[0] = positions0.copy()

    pos = positions0.copy()
    for t in range(T):
        pos = step_hardcore_sizeb(pos, t, beta, b, L, Tp, Nx, alpha, k=k)
        traj[t + 1] = pos

    return traj


# Run + plot 
def main():
    # Valgte initialer (b=2, L=80)
    positions0 = np.array([8 * i for i in range(N)], dtype=int)

    for i in range(N):
        for j in range(i + 1, N):
            if ring_distance(positions0[i], positions0[j], L) <= b:
                raise ValueError("Initial positions violate hard-core condition.")

    traj = simulate(positions0, T=T, beta=beta, b=b, L=L, Tp=Tp, Nx=Nx, alpha=alpha, k=1.0)

    plt.figure()
    for i in range(traj.shape[1]):
        plt.plot(traj[:, i], lw=1)
    plt.xlabel("time step")
    plt.ylabel("position (site index)")
    plt.title(f"Exercise 4a: flashing ratchet, beta*k=1000, b={b}, N={N}, L={L}")
    plt.show()
    
if __name__ == "__main__":
    main()







#oppgave 4b

import numpy as np
import matplotlib.pyplot as plt


beta = 1000.0
Nx = 100
Tp = 300
Ns = 10
alpha = 0.2
b = 20

L = Ns * Nx         
Nc = 100
T = Nc * Tp         #totalt steg

def sawtooth_V_at(x, Nx=100, alpha=0.2, k=1.0):
    xm = x % Nx
    m = int(round(alpha * Nx))
    m = max(1, min(Nx - 1, m))
    if xm < m:
        return k * (xm / m)
    else:
        return k * ((Nx - xm) / (Nx - m))
    
# sawtooth når den er på, for x=0,...,L-1
V_one_period = np.array([sawtooth_V_at(x, Nx=Nx, alpha=alpha, k=1.0) for x in range(Nx)])
V_on = np.tile(V_one_period, Ns)   # lengde L

def ring_distance(a, c, L):
    d = abs(a - c)
    return min(d, L - d)

def allowed_position(candidate, positions, i, b, L):
    if len(positions) <= 1:
        return True

    others = np.delete(positions, i)
    d = np.abs(others - candidate)
    d = np.minimum(d, L - d)
    return np.all(d >= b) #skal være raskere enn en for løkke


def transition_probabilities(x0, t, beta, L, Tp, V_on):
    xL = (x0 - 1) % L
    xR = (x0 + 1) % L

    on = (t % Tp) < (Tp // 2)
    if on:
        V0 = V_on[x0]
        VL = V_on[xL]
        VR = V_on[xR]
    else:
        V0 = VL = VR = 0.0

    wL = np.exp(-beta * (VL - V0))
    wR = np.exp(-beta * (VR - V0))
    w0 = 1.0
    Z = wL + w0 + wR
    
    return wL / Z, w0 / Z, wR / Z


def step_hardcore_sizeb_and_disp(positions, t, beta, b, L, Tp, V_on):
    positions = positions.copy()
    order = np.random.permutation(len(positions))

    disp_sum = 0  

    for i in order:
        x0 = positions[i]
        p_minus, p_0, p_plus = transition_probabilities(x0, t, beta, L, Tp, V_on)

        r = np.random.rand()
        if r <= p_minus:
            cand = (x0 - 1) % L
        elif r > 1.0 - p_plus:
            cand = (x0 + 1) % L
        else:
            cand = x0

        if cand != x0 and allowed_position(cand, positions, i, b=b, L=L):
            if cand == (x0 + 1) % L:
                disp_sum += 1
            elif cand == (x0 - 1) % L:
                disp_sum -= 1
            positions[i] = cand

    return positions, disp_sum


def run_current_for_density(Np, beta, b, L, Tp, Nc, V_on, seed=None):
    if seed is not None:
        np.random.seed(seed)

    positions = np.floor(np.linspace(0, L, Np, endpoint=False)).astype(int)
   
    for i in range(Np):
        for j in range(i + 1, Np):
            if ring_distance(positions[i], positions[j], L) < b:
                raise ValueError("Initial positions violate hard-core (distance < b).")

    J_cycles = []
    t_global = 0

    for _ in range(Nc):
        disp_cycle = 0
        
        for _ in range(Tp):
            positions, disp_step = step_hardcore_sizeb_and_disp(
                positions, t_global, beta, b, L, Tp, V_on
            )
            disp_cycle += disp_step
            t_global += 1

        
        J_cycles.append(disp_cycle / (Np * Tp))

    return float(np.mean(J_cycles)), float(np.std(J_cycles))

def main_4b():
    Np_max = L // b  
    Np_values = np.unique(np.linspace(1, Np_max, 12, dtype=int))

    rhos = (b * Np_values) / L
    Js = []
    Jerrs = []

   
    for Np, rho in zip(Np_values, rhos):
        J_mean, J_std = run_current_for_density(
            Np, beta=beta, b=b, L=L, Tp=Tp, Nc=Nc, V_on=V_on
        )
        Js.append(J_mean)
        Jerrs.append(J_std)

        print(f"Np={Np:2d}, rho={rho:.2f}, J={J_mean:.5f} ± {J_std:.5f}")

    Js = np.array(Js)
    Jerrs = np.array(Jerrs)

    plt.figure()
    plt.errorbar(rhos, Js, yerr=Jerrs, fmt='o-', capsize=3)
    plt.xlabel("density $\\rho$")
    plt.ylabel("cycle-averaged current $J$")
    plt.title("Exercise 4b: cycle-averaged current vs density")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main_4b()






