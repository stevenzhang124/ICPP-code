import numpy as np
from DAGs_Generator import gen_application
from Graph import gen_network
import random
import networkx as nx

random.seed(0)

# simulation parameters
# J = 20  # number of jobs
D = 50;  # number of edge devices
CCR = 0.5; # the communication to computation ratio
var = 0.2; # variance
B_avg = 20*10^6
PS_avg = 50*10^6
CL_avg = 300*10^3
max_iter = 30  # duplicate the testing for a more stable results


# Network Model
network, adjacency_matrix = gen_network(noD=D, var=var, B_avg=B_avg, PS_avg=PS_avg)


# Application model
# generate a random number to indicate how many tasks a job has
# no_task = random.randint(5,20)
no_task = 40
dep_data = CCR*((300*10^3)/(50*10^6))*B_avg

app_graph = gen_application(noT=no_task, CL_avg=CL_avg, dep_data=dep_data, var=var)

#print(list(app_graph.nodes(data=True)))


