import numpy as np
import matplotlib.pyplot as plt
from ODESolver import *
from SEIR import *

class RegionInteraction(Region):
    def __init__(self,name,S_0,E2_0,lat,long):
        Region.__init__(self,name,S_0,E2_0)
        self.lat = lat*np.pi/180
        self.long = long*np.pi/180

    def distance(self, other):
        R_earth = 64 #unit 10^5
        sin_v = np.sin(self.lat)*np.sin(other.lat)
        cos_v = np.cos(self.lat)*np.cos(other.lat)*np.cos(abs(self.long - other.long))
        if 0 <= (sin_v+cos_v) <=1:
            delta_sigma_ij = np.arccos(sin_v+cos_v)
            d = delta_sigma_ij * R_earth
            return d
class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia = 0.1,r_e2=1.25,lmbda_1=0.33, lmbda_2=0.5, p_a=0.4,mu=0.2):
        self.area_name = area_name
        self.region = region
        ProblemSEIR.__init__(self, region, beta, r_ia = 0.1, r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2)

    def get_population(self):
        s = 0
        for part in self.region:
            s += part.population
        return s

    def set_initial_condition(self):
        self.initial_condition = []
        for region in self.region:
            S_0 = region.S_0; E1_0 = region.E1_0; E2_0 = region.E2_0;
            I_0 = region.I_0; Ia_0 = region.Ia_0; R_0 = region.R_0;
            self.initial_condition += [S_0, E1_0, E2_0, I_0, Ia_0, R_0]

    def __call__(self, u, t):
        n = len(self.region)
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = []
        for i in range(n):
            S, E1, E2, I, Ia, R = SEIR_list[i]
            N = sum(SEIR_list[i])
            dS = 0
            for j in range(n):
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                N_j = self.get_population()
                dij = self.region[i].distance(self.region[j])
                dS += -self.r_ia *self.beta(t)*S*(Ia_other/N_j * np.exp(-dij)) - \
                        self.r_e2*self.beta(t)*S*(E2_other/N_j * np.exp(-dij))
            dS += -self.beta(t)*(S*I)/N
            dE1 = -dS - self.lmbda_1*E1
            dE2 = self.lmbda_1*(1-self.p_a)*E1 - self.lmbda_2*E2
            dI  = self.lmbda_2*E2 - self.mu*I
            dIa = self.lmbda_1*self.p_a*E1 - self.mu*Ia
            dR  = self.mu*(I + Ia)
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative

    def solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n)
        self.I = np.zeros(n)
        self.Ia = np.zeros(n)
        self.R = np.zeros(n)
        SEIR_list = [u[:, i:i+6] for i in range(0, n_reg*6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += part.S
            self.E1 += part.E1
            self.E2 += part.E2
            self.I += part.I
            self.Ia += part.Ia
            self.R += part.R

    def plot(self):
        plt.plot(self.t,self.S,label = "Susceptibles")
        plt.plot(self.t,self.I, label = "Infected")
        plt.plot(self.t,self.Ia,label = "Asymptomatic Infected")
        plt.plot(self.t,self.R,label = "Recovered")
        plt.xlabel("Time (days)")
        plt.ylabel("Population")
        plt.title(self.area_name)


if __name__ == '__main__':
    innlandet = RegionInteraction('Innlandet',S_0=371385, E2_0=0, \
    lat=60.7945,long=11.0680)
    oslo = RegionInteraction('Oslo',S_0=693494,E2_0=100, \
                            lat=59.9,long=10.8)
    print(oslo.distance(innlandet))
    print("---------------ProblemInteraction------------------------")

    problem = ProblemInteraction([oslo,innlandet],'Norway_east', beta=0.5)
    print(problem.get_population())
    problem.set_initial_condition()
    print(problem.initial_condition)
    u = problem.initial_condition
    print(problem(u,0))

    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve()
    problem.plot()
    plt.legend()
    plt.show()
