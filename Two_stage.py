'''
this algorithm is used to sovle the joint resource allocation and bandwidth and routing path optimzation problem
1. the first step is to determine the task allocation location
2. the second step is joint bandwidth allocation and routing optimization
	to solve the second subproblem, we relax it to a convex problem and solve it with a convex optimizer
'''
# must determine how the data is stored
import networkx as nx
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def maximum_trans_and_comp_time (G, task, network, node, job_assigned):
	predecessors = list(G.predecessors(task))
	trans_time_list = []
	for predecessor in predecessors:
		data_size = G.edges[predecessor, task]['weight']
		if job_assigned[predecessor] != node:
			# use the average bandwidth of all the routing paths
			# bandwidth = nx.dijkstra_path_length(network, source=job_assigned[predecessor], target=node)
			routing_paths = list(nx.all_simple_edge_paths(network, source=job_assigned[predecessor], target=node))
			routing_bandwidths = []
			for routing_path in routing_paths:
				bandwidths = []
				for edge in routing_path:
					bandwidth = network.edges[edge[0], edge[1]]['weight']
					bandwidths.append(bandwidth)

				routing_bandwidth = min(bandwidths)
				routing_bandwidths.append(routing_bandwidth)
			
			average_bandwidth = round(sum(routing_bandwidths) / len(routing_bandwidths), 2)
			trans_time = data_size / average_bandwidth
		else:
			trans_time = 0
		trans_time_list.append(trans_time)

	trans = max(trans_time_list)

	if G.nodes[task]['CL'] > network.nodes[node]['PS']:
		return False
	else:
		comp = G.nodes[task]['CL'] / network.nodes[node]['PS']

		return trans + comp


def task_allocation(job, network):
	'''
	Input: 
		job -> Graph: jobs to be scheduled, each job is a DAG
		arrive_time -> list: the arriving time of the jobs to be scheduled
		average_bandwidth -> float: the average bandwidth of the network
		average_ps -> float: the average computation power of the edge ndoes
	Output: the task allocation strategies
	'''
	# sorted_id = sorted(range(len(arrive_time)), key=lambda k: arrive_time[k], reverse=False)
	
	# for index in sorted_id:
	# 	job = jobs[index] # determine the job to be executed
	# 	job_assigned = []
	# 	for task in list(nx.bfs_tree(job, 'Start')): # schedule the task in a layered order
	# 		if task == 'Start':
	# 			job_assigned.append(job.nodes[task]['Source'])
	# 			continue
	# 		trans_and_comp_dict = {}
	# 		for node in network.nodes():
	# 			trans_and_comp = maximum_trans_and_comp_time(G, task, network, node, job_assigned)
	# 			trans_and_comp_dict[node] = trans_and_comp

	# 		target_node = min(trans_and_comp_dict, key=trans_and_comp_dict.get)
	# 		#update the processing power of edge node
	# 		network.nodes[target_node]['PS'] = network.nodes[target_node]['PS'] - G.nodes[task]['CL']

	# 		job_assigned.append(target_node)


	# return jobs_assigned
	job_assigned = []
	for task in list(nx.bfs_tree(job, 0)): # schedule the task in a layered order
		if task == 0:
			target_node = job.nodes[task]['Source']
			network.nodes[target_node]['PS'] = network.nodes[target_node]['PS'] - job.nodes[task]['CL']
			job_assigned.append(target_node)
			continue
		trans_and_comp_dict = {}
		for node in network.nodes():
			trans_and_comp = maximum_trans_and_comp_time(job, task, network, node, job_assigned)
			if trans_and_comp:
				trans_and_comp_dict[node] = trans_and_comp
			else:
				continue
		if trans_and_comp_dict:
			target_node = min(trans_and_comp_dict, key=trans_and_comp_dict.get)
			#update the processing power of edge node
			network.nodes[target_node]['PS'] = network.nodes[target_node]['PS'] - job.nodes[task]['CL']

			job_assigned.append(target_node)

		else:
			print("this job fail to be scheduled")
			return False   # this job cannot be scheduled to the edge nodes

	print(job_assigned)
	return job_assigned


def check_overlap_path(edge, all_routing_paths):
	index = []
	counter = 0
	edge_1 = edge
	edge_2 = (edge[1], edge[0])
	for i in range(len(all_routing_paths)): # for the routing paths of each flow
		for j in range(len(all_routing_paths[i])):
			if edge_1 in all_routing_paths[i][j] or edge_2 in all_routing_paths[i][j]:
				index.append(counter+j)
			
		counter = counter + len(all_routing_paths[i])
	return index


def joint_bandwidth_and_routing(network, job_assigned, job):
	'''
	consider the single job for now
	Input: the flows of the tasks <source, destination, datasize, routing_paths>, the network topology and bandwidth
	Outpt:
	'''
	#generate the <source, destination, datasize> pair

	#generate the matrix required for calling cvxopt
	# find how many dataflow we have Q
	# through the source and destination find the possible routing path
	# calculate the total number of routing path of all flows K
	# we have Q + K + 1 variables and 
	# number of constraints: the links that involved + Q + K;  equations Q

	# Find Q
	flows = []
	all_routing_paths = []
	for i, task in enumerate(list(nx.bfs_tree(job, 0))):
		if task == 0:
			continue
		dest_node = job_assigned[i]
		predecessors = list(job.predecessors(task))
		for predecessor in predecessors:
			if predecessor == 0:
				index = 0
				source_node = job_assigned[index]
			else:
				index = predecessor
				source_node = job_assigned[index]

			if dest_node != source_node:
				datasize = job.edges[predecessor, task]['weight']
				routing_paths = list(nx.all_simple_edge_paths(network, source=source_node, target=dest_node))
				# [[(0, 1), (1, 2), (2, 3)], [(0, 1), (1, 3)], [(0, 2), (2, 1), (1, 3)], [(0, 2), (2, 3)], [(0, 3)]]
				all_routing_paths.append(routing_paths)
				flow = (source_node, dest_node, datasize, routing_paths)
				flows.append(flow)
	
	Q = len(flows)
	K = 0
	edges = []
	# Find K and number of links involved
	for routing_paths in all_routing_paths:
		K = K + len(routing_paths)
		for path in routing_paths:
			for edge in path:
				edges.append(edge)
	edges = list(set(edges))
	# remove those duplicated edges
	temp_graph = nx.Graph()
	temp_graph.add_edges_from(edges)
	edges = list(temp_graph.edges())

	link_num = len(edges)

	#construct the c, G, h, A, b matrix
	objective_function = []
	for i in range(Q + K + 1):
		if i != (Q+K):
			objective_function.append(0.)
		else:
			objective_function.append(1.)

	c = matrix(objective_function)

	# G
	# the inequation  rows x columns   (link_num + Q + K) x (Q + K + 1)
	rows = link_num + Q + K
	columns = Q + K + 1
	G_temp = [[0. for i in range(columns)] for j in range(rows)]
	
	for i in range(rows):
		if i < Q + K:
			G_temp[i][i] = -1.
		else:
			# find the returned j
			temp_column = check_overlap_path(edges[i-Q-K], all_routing_paths)
			for j in temp_column:
				G_temp[i][j] = 1.

			G_temp[i][-1] = network.edges[edges[i-Q-K][0], edges[i-Q-K][1]]['weight'] * -1.

	G_temp_transpose = list(map(list, zip(*G_temp)))
	G = matrix(G_temp_transpose)

	# h
	h_temp = [0. for i in range(rows)]
	for i in range(rows):
		if i >= K and i < K+Q:
			h_temp[i] = flows[i-K][2] * -1.

	h = matrix(h_temp)



	# A
	A_rows = Q
	A_columns = Q + K + 1
	A_temp = [[0. for i in range(A_columns)] for j in range(A_rows)]
	
	counter = 0
	for i in range(A_rows):
		for j in range(len(flows[i])):
			A_temp[i][counter+j] = 1.

		counter = counter + len(flows[i])
		A_temp[i][K+i] = -1.

	A_temp_transpose = list(map(list, zip(*A_temp)))

	A = matrix(A_temp_transpose)

	# b
	b_temp = [0. for i in range(A_rows)]
	b = matrix(b_temp)

	sol = solvers.lp(c, G, h, A, b)

	print(sol['x'])	
	print(sol['primal objective'])

	print(list(sol['x']))
	print([round(x, 2) for x in list(sol['x'])])
	solution = [round(x, 2) for x in list(sol['x'])]

	################## Finish Solve the Linear Programming Problem ####################################

	# Determine the routing path X_{i}^{k}
	counter = 0
	routing_path_solutions = []
	for flow in flows:
		# index_range = [ counter+i for i in range(len(flow[3])) ]
		index = solution.index(max(solution[counter:(counter+len(flow[3]))]))
		routing_path_solution = flow[3][index-counter]
		print(routing_path_solution)
		routing_path_solutions.append(routing_path_solution)
		counter = counter + len(flow[3])

	# Determine the bandwidth allocation b_{i}, once the routing path determined, there is analytical solution
	# determine the edges involved, 
	routing_bandwidth_solutions = []
	trans_times = []
	for i in range(len(routing_path_solutions)):
		flow = []
		bandwidths = []
		for edge in routing_path_solutions[i]:
			total_datasize = 0
			for j in range(len(routing_path_solutions)):
				if edge in routing_path_solutions[j] or (edge[1], edge[0]) in routing_path_solutions[j]:
					total_datasize = total_datasize + flows[j][2]	
			
			bandwidth = round(flows[i][2] / total_datasize * network.edges[edge[0], edge[1]]['weight'], 2)
			bandwidths.append(bandwidth)
			
		min_bandwidth = min(bandwidths)

		trans_time = flows[i][2] / min_bandwidth
		trans_times.append(trans_time)
		
		for edge in routing_path_solutions[i]:
			edge_temp = (edge, min_bandwidth)
			flow.append(edge_temp)

		routing_bandwidth_solutions.append(flow)

	print(routing_bandwidth_solutions)
	print(trans_times)

	print("the maximum trans_time is ", max(trans_times))


	# update the remaining computation and network resources
	for i in range(len(routing_bandwidth_solutions)):
		for (edge, bandwidth) in routing_bandwidth_solutions[i]:
			network.edges[edge[0], edge[1]]['weight'] = network.edges[edge[0], edge[1]]['weight'] - bandwidth

	pos = nx.spring_layout(network)
	nx.draw(network, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(network, 'weight')
	nx.draw_networkx_edge_labels(network, pos, edge_labels = edge_labels)

	plt.show()


def test():
	######################### Test Function joint_bandwidth_and_routing #################################
	# generate network
	# links = [('A', 'B', 10), ('B', 'F', 20), ('F', 'D', 50), ('D', 'C', 15), ('C', 'A', 5), ('B', 'E', 7), ('E', 'D', 4)]
	# G = nx.Graph()
	# G.add_weighted_edges_from(links)

	# pos = nx.spring_layout(G)
	# nx.draw(G, pos, with_labels=True)

	# edge_labels = nx.get_edge_attributes(G, 'weight')
	# nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	# plt.show()

	# # generate application
	
	# app_links = [(0, 1, 50), (0, 2, 100), (2, 3, 30), (3, 4, 10)]
	# app = nx.DiGraph()
	# app.add_weighted_edges_from(app_links)

	# pos = nx.spring_layout(app)
	# nx.draw(app, pos, with_labels=True)

	# edge_labels = nx.get_edge_attributes(app, 'weight')
	# nx.draw_networkx_edge_labels(app, pos, edge_labels = edge_labels)

	# plt.show()

	# # set the task allocation strategies
	# task_list = list(nx.bfs_tree(app, 0))
	# # print(task_list)  [0, 1, 2, 3, 4]
	# job_assigned = ['A', 'E', 'F', 'D', 'D']

	# return G, app, job_assigned


	links = [('A', 'B', 10), ('B', 'F', 20), ('F', 'D', 50), ('D', 'C', 15), ('C', 'A', 5), ('B', 'E', 7), ('E', 'D', 4)]
	G = nx.Graph()
	G.add_weighted_edges_from(links)

	# add computation capability
	G.nodes['A']['PS'] = 100
	G.nodes['B']['PS'] = 50
	G.nodes['C']['PS'] = 200
	G.nodes['D']['PS'] = 150
	G.nodes['E']['PS'] = 30
	G.nodes['F']['PS'] = 300

	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(G, 'weight')
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	plt.show()

	# generate application
	
	app_links = [(0, 1, 50), (0, 2, 100), (2, 3, 30), (3, 4, 10)]
	app = nx.DiGraph()
	app.add_weighted_edges_from(app_links)

	app.nodes[0]['Source'] = 'A'
	# add computation workload
	app.nodes[0]['CL'] = 50
	app.nodes[1]['CL'] = 150
	app.nodes[2]['CL'] = 200
	app.nodes[3]['CL'] = 90
	app.nodes[4]['CL'] = 100

	pos = nx.spring_layout(app)
	nx.draw(app, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(app, 'weight')
	nx.draw_networkx_edge_labels(app, pos, edge_labels = edge_labels)

	plt.show()

	return G, app

	



if __name__ == '__main__':
	network, job = test()
	job_assigned = task_allocation(job, network)
	# network, job, job_assigned= test()
	joint_bandwidth_and_routing(network, job_assigned, job)

