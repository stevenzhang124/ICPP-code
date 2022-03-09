import numpy as np
from Attribute_recognition import gen_application
from Graph import gen_network
import random
import networkx as nx

from Baselines import least_request_priority, balanced_resource_allocation, task_allocation_no_routing_bandwidth_1, task_allocation_no_routing_bandwidth_2
from Two_stage import task_allocation, joint_bandwidth_and_routing
import copy
import time

random.seed(0)

# simulation parameters
# J = 20  # number of jobs
D = 10;  # number of edge devices
var = 0.2; # variance
B_avg = 1  # MB/s
PS_avg = 5000
max_iter = 30  # duplicate the testing for a more stable results

print("Number of nodes are ", D)
# Network Model
network_origin = gen_network(noD=D, var=var, B_avg=B_avg, PS_avg=PS_avg)

for source in range(D-1):
	# Application model
	app_graph = gen_application(noD=D, source=source)

	#print(list(app_graph.nodes(data=True)))

	############################### call the functions (proposed method and baselines) #################################################

	# baseline 1
	network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
	start = time.time()
	results = least_request_priority(job, network)
	end = time.time()
	if results:
		print("Baseline 1:")
		print("Target Node: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
		print("Baseline 1 consumes ", end - start)

	# baseline 2
	network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
	start = time.time()
	results = balanced_resource_allocation(job, network)
	end = time.time()
	if results:
		print("Baseline 2:")
		print("Target Node: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
		print("Baseline 2 consumes ", end - start)

	# baseline 3
	network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
	start = time.time()
	results = task_allocation_no_routing_bandwidth_1(job, network)
	end = time.time()
	if results:
		print("Baseline 3:")
		print("Job Assigned: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
		print("Baseline 3 consumes ", end - start)

	# baseline 4
	network, job = copy.deepcopy(network_origin), copy.deepcopy(app_graph)
	start = time.time()
	results = task_allocation_no_routing_bandwidth_2(job, network)
	end = time.time()
	if results:
		print("Baseline 4:")
		print("Job Assigned: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
		print("Baseline 4 consumes ", end - start)

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
		print("Proposed solution consumes ", end - start)

	print("--------------------------------------------------------------------------")
