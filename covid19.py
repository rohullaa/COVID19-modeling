import numpy as np
import matplotlib.pyplot as plt
from ODESolver import *
from SEIR import *
from SEIR_interaction import *

def readFile(filename):
    counties_info = []
    region_info = []
    with open(filename,'r') as infile:
        for line in infile:
            a = line.split(';')
            counties_info += [[a[1],a[2][1:],a[3],a[4],a[5][0:-1]]]
    return counties_info

def covid19_Norway(beta, filename, num_days, dt):
    counties_info = readFile(filename)
    regions = [counties_info[i][0] for i in range(0, len(counties_info), 1)]
    S_0s = [float(counties_info[i][1]) for i in range(0, len(counties_info), 1)]
    E2_0s = [float(counties_info[i][2]) for i in range(0, len(counties_info), 1)]
    lats = [float(counties_info[i][3]) for i in range(0, len(counties_info), 1)]
    longs = [float(counties_info[i][4]) for i in range(0, len(counties_info), 1)]

    Viken = RegionInteraction(regions[0],S_0s[0], E2_0s[0], lats[0],longs[0])
    Oslo = RegionInteraction(regions[1],S_0s[1], E2_0s[1], lats[1],longs[1])
    Innlandet = RegionInteraction(regions[2],S_0s[2], E2_0s[2], lats[2],longs[2])
    Vestfold = RegionInteraction(regions[3],S_0s[3], E2_0s[3], lats[3],longs[3])
    Agder = RegionInteraction(regions[4],S_0s[4], E2_0s[4], lats[4],longs[4])
    Rogaland = RegionInteraction(regions[5],S_0s[5], E2_0s[5], lats[5],longs[5])
    Vestland = RegionInteraction(regions[6],S_0s[6], E2_0s[6], lats[6],longs[6])
    Møre_og_Romsdal = RegionInteraction(regions[7],S_0s[7], E2_0s[7], lats[7],longs[7])
    Trondelag = RegionInteraction(regions[8],S_0s[8], E2_0s[8], lats[8],longs[8])
    Nordland = RegionInteraction(regions[9],S_0s[9], E2_0s[9], lats[9],longs[9])
    Troms_og_Finnmark = RegionInteraction(regions[10],S_0s[10], E2_0s[10], lats[10],longs[10])

    region_names = [Viken,Oslo,Innlandet,Vestfold,Agder,Rogaland,Vestland,Møre_og_Romsdal,Trondelag,Nordland,Troms_og_Finnmark]

    problem = ProblemInteraction(region_names,'Counties of Norway',beta)
    solver = SolverSEIR(problem,T=num_days,dt=dt)
    solver.solve()
    plt.figure(figsize=(13, 8)) # set figsize
    index = 1
    for part in problem.region:
        plt.subplot(4,3,index)
        part.plot()
        index += 1
    plt.subplot(4,3, index)
    plt.subplots_adjust(hspace = 0.75, wspace=0.5)
    problem.plot()

    plt.legend()
    plt.show()

covid19_Norway(0.5,"fylker.txt",150,1.0 )

def test_reading():
    #testing files fylker.txt with fylker1.txt
    counties_info = readFile("fylker.txt")
    counties_info1 = readFile("fylker1.txt")

    for i in range(0,len(counties_info1)):
        if counties_info1[i] in counties_info:
            return
        else:
            print("Error")

test_reading()
