import numpy as np
from DAGs import gen_application
from Graph import gen_network
import random
import networkx as nx

from Baselines import least_request_priority, balanced_resource_allocation, task_allocation_no_routing_bandwidth_1, task_allocation_no_routing_bandwidth_2
from Two_stage import task_allocation, joint_bandwidth_and_routing
import copy

random.seed(0)

# simulation parameters
# J = 20  # number of jobs
D = 20;  # number of edge devices
CCR = 0.5; # the communication to computation ratio
var = 0.2; # variance
B_avg = 2000
PS_avg = 5000
CL_avg = 30
max_iter = 30  # duplicate the testing for a more stable results


# Network Model
network_origin = gen_network(noD=D, var=var, B_avg=B_avg, PS_avg=PS_avg)


# Application model
# generate a random number to indicate how many tasks a job has
# no_task = random.randint(5,20)
no_task = 20
dep_data = CCR*(CL_avg/PS_avg)*B_avg

app_graph = gen_application(noT=no_task, CL_avg=CL_avg, dep_data=dep_data, var=var)

#print(list(app_graph.nodes(data=True)))

############################### call the functions (proposed method and baselines) #################################################

# baseline 1
network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
results = least_request_priority(job, network)
if results:
	print("Baseline 1:")
	print("Target Node: ", results[0])
	print("Job Completion Time: ", results[1])
	print("Throughput: ", results[2])
# baseline 2
network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
results = balanced_resource_allocation(job, network)
if results:
	print("Baseline 2:")
	print("Target Node: ", results[0])
	print("Job Completion Time: ", results[1])
	print("Throughput: ", results[2])
# baseline 3
network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
results = task_allocation_no_routing_bandwidth_1(job, network)
if results:
	print("Baseline 3:")
	print("Job Assigned: ", results[0])
	print("Job Completion Time: ", results[1])
	print("Throughput: ", results[2])
# baseline 4
network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
results = task_allocation_no_routing_bandwidth_2(job, network)
if results:
	print("Baseline 4:")
	print("Job Assigned: ", results[0])
	print("Job Completion Time: ", results[1])
	print("Throughput: ", results[2])

# proposed solution
print("Proposed Solution:")
network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
job_assigned = task_allocation(job, network)
if job_assigned:
	print("Job Assigned: ", job_assigned)
	# network, job, job_assigned= test()
	results = joint_bandwidth_and_routing(network, job_assigned, job)
	print("Job Completion Time: ", results[0])
	print("Throughput: ", results[1])
