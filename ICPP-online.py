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
# D = 10;  # number of edge devices
var = 0.2; # variance
# B_avg = 1  # MB/s
PS_avg = 5000
max_iter = 30  # duplicate the testing for a more stable results
N_jobs = 20

bw_list = [1, 5, 10, 15, 20]
D_list = [10, 50]
# D_list = [10, 20, 30, 40, 50, 75, 100]
# N_jobs_list = [20, 30, 40, 50, 60, 70, 80, 90, 100]


for B_avg in bw_list:
	for D in D_list:
		print("Average bandwidth is ", B_avg)
		print("Number of nodes are ", D)
		print("Number of jobs are ", N_jobs)
		# Network Model
		network_origin = gen_network(noD=D, var=var, B_avg=B_avg, PS_avg=PS_avg)

		# Application model
		app_graphs = []
		for job_num in range(N_jobs):
			app_graph = gen_application(noD=D)
			app_graphs.append(app_graph)

		#print(list(app_graph.nodes(data=True)))

		############################### call the functions (proposed method and baselines) #################################################

		# baseline 1
		print("Baseline 1:")
		network= copy.deepcopy(network_origin)
		failed_job_time = []
		Throughputs = []
		start = time.time()
		for (i, job) in enumerate(app_graphs):
			results = least_request_priority(job, network)
			if results:
				print("Target Node: ", results[0])
				print("Job Completion Time: ", results[1])
				print("Throughput: ", results[2])
				Throughputs.append(results[2])
			else:
				print("Job " + str(i) + " fail to be scheduled")
				failed_job_time.append(time.time())
		end = time.time()
		total_failed_time = 0
		for failed_job in failed_job_time:
			total_failed_time = total_failed_time + end - failed_job
		print("Baseline 1 consumes ", end - start + total_failed_time)
		print("The average Throughput is", sum(Throughputs) / len(Throughputs))

		print("--------------------------------------------------------------------------")

		# baseline 2
		print("Baseline 2:")
		network= copy.deepcopy(network_origin)
		failed_job_time = []
		Throughputs = []
		start = time.time()
		for (i, job) in enumerate(app_graphs):
			results = balanced_resource_allocation(job, network)
			if results:
				print("Target Node: ", results[0])
				print("Job Completion Time: ", results[1])
				print("Throughput: ", results[2])
				Throughputs.append(results[2])
			else:
				print("Job " + str(i) + " fail to be scheduled")
				failed_job_time.append(time.time())
		end = time.time()
		total_failed_time = 0
		for failed_job in failed_job_time:
			total_failed_time = total_failed_time + end - failed_job
		print("Baseline 2 consumes ", end - start + total_failed_time)
		print("The average Throughput is", sum(Throughputs) / len(Throughputs))

		print("--------------------------------------------------------------------------")

		# baseline 3
		print("Baseline 3:")
		network = copy.deepcopy(network_origin)
		failed_job_time = []
		Throughputs = []
		start = time.time()
		for (i, job) in enumerate(app_graphs):
			results = task_allocation_no_routing_bandwidth_1(job, network)
			if results:
				print("Job Assigned: ", results[0])
				print("Job Completion Time: ", results[1])
				print("Throughput: ", results[2])
				Throughputs.append(results[2])
			else:
				print("Job " + str(i) + " fail to be scheduled")
				failed_job_time.append(time.time())
		end = time.time()
		total_failed_time = 0
		for failed_job in failed_job_time:
			total_failed_time = total_failed_time + end - failed_job
		print("Baseline 3 consumes ", end - start + total_failed_time)
		print("The average Throughput is", sum(Throughputs) / len(Throughputs))

		print("--------------------------------------------------------------------------")

		# baseline 4
		print("Baseline 4:")
		network = copy.deepcopy(network_origin)
		failed_job_time = []
		Throughputs = []
		start = time.time()
		for (i, job) in enumerate(app_graphs):
			results = task_allocation_no_routing_bandwidth_2(job, network)
			if results:
				print("Job Assigned: ", results[0])
				print("Job Completion Time: ", results[1])
				print("Throughput: ", results[2])
				Throughputs.append(results[2])
			else:
				print("Job " + str(i) + " fail to be scheduled")
				failed_job_time.append(time.time())
		end = time.time()
		total_failed_time = 0
		for failed_job in failed_job_time:
			total_failed_time = total_failed_time + end - failed_job
		print("Baseline 4 consumes ", end - start + total_failed_time)
		print("The average Throughput is", sum(Throughputs) / len(Throughputs))

		print("--------------------------------------------------------------------------")

		# proposed solution
		print("Proposed Solution:")
		network = copy.deepcopy(network_origin)
		failed_job_time = []
		Throughputs = []
		start = time.time()
		for (i, job) in enumerate(app_graphs):
			st = time.time()
			job_assigned = task_allocation(job, network)
			et = time.time()
			print("Assign tasks consume", st - et)
			if job_assigned:
				print("Job Assigned: ", job_assigned)
				# network, job, job_assigned= test()
				results = joint_bandwidth_and_routing(network, job_assigned, job)
				print("Job Completion Time: ", results[0])
				print("Throughput: ", results[1])
				Throughputs.append(results[1])
			else:
				print("Job " + str(i) + " fail to be scheduled")
				failed_job_time.append(time.time())
		end = time.time()
		total_failed_time = 0
		for failed_job in failed_job_time:
			total_failed_time = total_failed_time + end - failed_job
		print("Proposed solution consumes ", end - start + total_failed_time)
		print("The average Throughput is", sum(Throughputs) / len(Throughputs))

		print("--------------------------------------------------------------------------")


# for B_avg in bw_list:
# 	for D in D_list:
# 		for N_jobs in N_jobs_list:
# 			if D >= N_jobs:

# 				print("Average bandwidth is ", B_avg)
# 				print("Number of nodes are ", D)
# 				print("Number of jobs are ", N_jobs)
# 				# Network Model
# 				network_origin = gen_network(noD=D, var=var, B_avg=B_avg, PS_avg=PS_avg)

# 				# Application model
# 				app_graphs = []
# 				for job_num in range(N_jobs):
# 					app_graph = gen_application(noD=D)
# 					app_graphs.append(app_graph)

# 				#print(list(app_graph.nodes(data=True)))

# 				############################### call the functions (proposed method and baselines) #################################################

# 				# baseline 1
# 				print("Baseline 1:")
# 				network= copy.deepcopy(network_origin)
# 				failed_job_time = []
# 				Throughputs = []
# 				start = time.time()
# 				for (i, job) in enumerate(app_graphs):
# 					results = least_request_priority(job, network)
# 					if results:
# 						print("Target Node: ", results[0])
# 						print("Job Completion Time: ", results[1])
# 						print("Throughput: ", results[2])
# 						Throughputs.append(results[2])
# 					else:
# 						print("Job " + str(i) + " fail to be scheduled")
# 						failed_job_time.append(time.time())
# 				end = time.time()
# 				total_failed_time = 0
# 				for failed_job in failed_job_time:
# 					total_failed_time = total_failed_time + end - failed_job
# 				print("Baseline 1 consumes ", end - start + total_failed_time)
# 				print("The average Throughput is", sum(Throughputs) / len(Throughputs))

# 				print("--------------------------------------------------------------------------")

# 				# baseline 2
# 				print("Baseline 2:")
# 				network= copy.deepcopy(network_origin)
# 				failed_job_time = []
# 				Throughputs = []
# 				start = time.time()
# 				for (i, job) in enumerate(app_graphs):
# 					results = balanced_resource_allocation(job, network)
# 					if results:
# 						print("Target Node: ", results[0])
# 						print("Job Completion Time: ", results[1])
# 						print("Throughput: ", results[2])
# 						Throughputs.append(results[2])
# 					else:
# 						print("Job " + str(i) + " fail to be scheduled")
# 						failed_job_time.append(time.time())
# 				end = time.time()
# 				total_failed_time = 0
# 				for failed_job in failed_job_time:
# 					total_failed_time = total_failed_time + end - failed_job
# 				print("Baseline 2 consumes ", end - start + total_failed_time)
# 				print("The average Throughput is", sum(Throughputs) / len(Throughputs))

# 				print("--------------------------------------------------------------------------")

# 				# baseline 3
# 				print("Baseline 3:")
# 				network = copy.deepcopy(network_origin)
# 				failed_job_time = []
# 				Throughputs = []
# 				start = time.time()
# 				for (i, job) in enumerate(app_graphs):
# 					results = task_allocation_no_routing_bandwidth_1(job, network)
# 					if results:
# 						print("Job Assigned: ", results[0])
# 						print("Job Completion Time: ", results[1])
# 						print("Throughput: ", results[2])
# 						Throughputs.append(results[2])
# 					else:
# 						print("Job " + str(i) + " fail to be scheduled")
# 						failed_job_time.append(time.time())
# 				end = time.time()
# 				total_failed_time = 0
# 				for failed_job in failed_job_time:
# 					total_failed_time = total_failed_time + end - failed_job
# 				print("Baseline 3 consumes ", end - start + total_failed_time)
# 				print("The average Throughput is", sum(Throughputs) / len(Throughputs))

# 				print("--------------------------------------------------------------------------")

# 				# baseline 4
# 				print("Baseline 4:")
# 				network = copy.deepcopy(network_origin)
# 				failed_job_time = []
# 				Throughputs = []
# 				start = time.time()
# 				for (i, job) in enumerate(app_graphs):
# 					results = task_allocation_no_routing_bandwidth_2(job, network)
# 					if results:
# 						print("Job Assigned: ", results[0])
# 						print("Job Completion Time: ", results[1])
# 						print("Throughput: ", results[2])
# 						Throughputs.append(results[2])
# 					else:
# 						print("Job " + str(i) + " fail to be scheduled")
# 						failed_job_time.append(time.time())
# 				end = time.time()
# 				total_failed_time = 0
# 				for failed_job in failed_job_time:
# 					total_failed_time = total_failed_time + end - failed_job
# 				print("Baseline 4 consumes ", end - start + total_failed_time)
# 				print("The average Throughput is", sum(Throughputs) / len(Throughputs))

# 				print("--------------------------------------------------------------------------")

# 				# proposed solution
# 				print("Proposed Solution:")
# 				network = copy.deepcopy(network_origin)
# 				failed_job_time = []
# 				Throughputs = []
# 				start = time.time()
# 				for (i, job) in enumerate(app_graphs):
# 					job_assigned = task_allocation(job, network)
# 					if job_assigned:
# 						print("Job Assigned: ", job_assigned)
# 						# network, job, job_assigned= test()
# 						results = joint_bandwidth_and_routing(network, job_assigned, job)
# 						print("Job Completion Time: ", results[0])
# 						print("Throughput: ", results[1])
# 						Throughputs.append(results[1])
# 					else:
# 						print("Job " + str(i) + " fail to be scheduled")
# 						failed_job_time.append(time.time())
# 				end = time.time()
# 				total_failed_time = 0
# 				for failed_job in failed_job_time:
# 					total_failed_time = total_failed_time + end - failed_job
# 				print("Proposed solution consumes ", end - start + total_failed_time)
# 				print("The average Throughput is", sum(Throughputs) / len(Throughputs))

# 				print("--------------------------------------------------------------------------")
