### oppgave 4a



import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Parameter (gitt)
# -------------------------
beta = 1000.0          # slik at beta*k = 1000 when k=1
Nx = 20
alpha = 0.2
Tp = 40

n_peaks = 4
L = Nx * n_peaks       # 80 sites total
n_cycles = 5
T = n_cycles * Tp      # 200 steps

# Choose (as asked)
N = 10                 # nummer of particles
b = 2                  # particle size (exclusion distance)


# -------------------------
# i) Ratchet potential + flashing
# -------------------------
def sawtooth_V_at(x, Nx=20, alpha=0.2, k=1.0):
    """
    Sawtooth potential over one period Nx, extended periodically.
    Returns V(x) for integer x (site index).
    """
    xm = x % Nx
    m = int(round(alpha * Nx))
    m = max(1, min(Nx - 1, m))

    # Piecewise linear "tooth"
    if xm < m:
        return k * (xm / m)                         # ramp up
    else:
        return k * ((Nx - xm) / (Nx - m))          # ramp down


def flashing_V_at(x, t, Tp=40, Nx=20, alpha=0.2, k=1.0):
    """
    Flashing: ON for half period, OFF for half period.
    """
    on = (t % Tp) < (Tp // 2)
    if on:
        return sawtooth_V_at(x, Nx=Nx, alpha=alpha, k=k)
    else:
        return 0.0


# -------------------------
# ii) Random walk step with:
# - periodic BC
# - hard core with size b (no distance <= b)
# -------------------------
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
    """
    Heat-bath style probabilities based on local potential difference.
    If you must use a specific Eq.(8) from the compendium, replace this function
    with your Eq.(8) implementation.

    Moves: left, stay, right.
    """
    xL = (x0 - 1) % L
    xR = (x0 + 1) % L

    V0 = flashing_V_at(x0, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)
    VL = flashing_V_at(xL, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)
    VR = flashing_V_at(xR, t, Tp=Tp, Nx=Nx, alpha=alpha, k=k)

    # Boltzmann weights for attempting left/right relative to current
    wL = np.exp(-beta * (VL - V0))
    wR = np.exp(-beta * (VR - V0))
    w0 = 1.0

    Z = wL + w0 + wR
    p_minus = wL / Z
    p_plus = wR / Z
    p_0 = w0 / Z
    return p_minus, p_0, p_plus


def step_hardcore_sizeb(positions, t, beta, b, L, Tp, Nx, alpha, k=1.0):
    """
    One time step: random sequential update (each particle tries once),
    periodic boundary conditions, and size-b hard-core repulsion.
    """
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


# -------------------------
# Run + plot (as asked)
# -------------------------
def main():
    # Choose initial positions (safe with b=2 on L=80)
    positions0 = np.array([8 * i for i in range(N)], dtype=int)

    # Sanity check: ensure initial config satisfies distance > b
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

