import numpy as np
import matplotlib.pyplot as plt
from ODESolver import *

def SEIR(u,t):
    beta =0.5; r_ia =0.1; r_e2=1.25;
    lmbda_1=0.33; lmbda_2=0.5; p_a=0.4; mu=0.2;

    S, E1, E2, I, Ia, R = u
    N = sum(u)
    dS  = -beta*S*I/N - r_ia*beta*S*Ia/N - r_e2*beta*S*E2/N
    dE1 = beta*S*I/N + r_ia*beta*S*Ia/N + r_e2*beta*S*E2/N - lmbda_1*E1
    dE2 = lmbda_1*(1-p_a)*E1 - lmbda_2*E2
    dI  = lmbda_2*E2 - mu*I
    dIa = lmbda_1*p_a*E1 - mu*Ia
    dR  = mu*(I + Ia)
    return [dS, dE1, dE2, dI, dIa, dR]

def test_SEIR():
    tol = 1e-10;
    t = 0; u = [1,1,1,1,1,1]
    computed = SEIR(u,t)
    exact = [-0.19583333333333333, -0.13416666666666668, -0.302, 0.3, -0.068, 0.4];

    for i in range(0,len(computed)):
        msg = f'Error: computed[i] = {computed[i]} != exact[i] = {exact[i]}'
        assert abs(computed[i]-exact[i]) < tol, msg

test_SEIR()

def solve_SEIR(T,dt,S_0,E2_0):
    N = int(T/dt)
    time = np.linspace(0,T,N)
    U0 = [S_0, 0, E2_0, 0, 0, 0]

    solver = RungeKutta4(SEIR)
    solver.set_initial_condition(U0)
    u, t = solver.solve(time)

    return u ,t

def plot_SEIR(u,t):
    S = u[:,0]
    I = u[:,3]
    Ia = u[:,4]
    R = u[:,5]

    plt.plot(t,S,label = "S")
    plt.plot(t,I, label = "I")
    plt.plot(t,Ia,label = "Ia")
    plt.plot(t,R,label = "R")
    plt.legend()
    plt.show()

S_0=5e6; E2_0=100; T = 100; dt = 1.0;
u,t = solve_SEIR(T,dt, S_0,E2_0)
plot_SEIR(u,t)


# running example:
# python3 seir_func.py
# shows the plot of S,I,Ia and R.
