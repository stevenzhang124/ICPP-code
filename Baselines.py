'''
baseline 1: least_request_priority
baeline 2: balanced_resource_allocation

baseline 3: task_allocation_no_routing_bandwidth_1
baseline 4: task_allocation_no_routing_bandwidth_2

still have to calcualte the job completion time
'''
import copy
import networkx as nx
import matplotlib.pyplot as plt
from Two_stage import maximum_trans_and_comp_time

def check_shorest_path(network, source, dest):
	'''
	the shorest path between two node should be the path with max transmission bandwidth
	'''
	routing_paths = list(nx.all_simple_edge_paths(network, source=source, target=dest))
	routing_bandwidths = []
	for routing_path in routing_paths:
		bandwidths = []
		for edge in routing_path:
			bandwidth = network.edges[edge[0], edge[1]]['weight']
			bandwidths.append(bandwidth)

		routing_bandwidth = min(bandwidths)
		routing_bandwidths.append(routing_bandwidth)
	
	shorest_path_bandwidth = max(routing_bandwidths)
	shorest_path = routing_paths[routing_bandwidths.index(shorest_path_bandwidth)]

	return shorest_path_bandwidth, shorest_path


def least_request_priority(job, network):
	'''
	Input:
	Output: where to allocate the job
	'''
	# find the node with the least resource consumption (node with the most powerful resource)
	node_avail_resource = {}
	for node in network.nodes():
		node_avail_resource[node] = network.nodes[node]['resource']

	target_node = max(node_avail_resource, key=node_avail_resource.get)

	if job.nodes['Source']["total_request_resource"] < node_avail_resource[target_node]:
		network.nodes[target_node]['resource'] = network.nodes[target_node]['resource'] - job.nodes['Source']["total_request_resource"]
		# calculate the job completion time
		if target_node == job.nodes['Source']['source']:
			trans_time = 0
		else:
			shorest_path_bandwidth, shorest_path = check_shorest_path(network, job.nodes['Source']['source'], target_node)
			# print("the bandwidth is ", shorest_path_bandwidth)
			# print("the shorest path ", shorest_path)
			trans_time = job.nodes['Source']['Source_datasize'] / shorest_path_bandwidth

		comp_time = job.nodes['Source']['total_workload'] / network.nodes[target_node]['PS']
		job_completion_time = comp_time + trans_time
		job_throughput = round(1/max(comp_time, trans_time), 2)

		return target_node, job_completion_time, job_throughput
	else:
		# print("this job fail to be scheduled")
		return False


def balanced_resource_allocation(job, network):

	node_balance_ratio = {}
	for node in network.nodes():
		if job.nodes['Source']['total_request_resource'] < network.nodes[node]['resource']: # select nodes with sufficient resources
			node_balance_ratio[node] = (network.nodes[node]['resource'] - job.nodes['Source']['total_request_resource']) / network.nodes[node]['max_resource']
	
	if not node_balance_ratio:
		# print("this job fail to be scheduled")
		return False

	target_node = max(node_balance_ratio, key=node_balance_ratio.get)
	# calculate the job completion time
	if target_node == job.nodes['Source']['source']:
			trans_time = 0
	else:
		shorest_path_bandwidth, shorest_path = check_shorest_path(network, job.nodes['Source']['source'], target_node)
		# print("the bandwidth is ", shorest_path_bandwidth)
		# print("the shorest path ", shorest_path)
		trans_time = job.nodes['Source']['Source_datasize'] / shorest_path_bandwidth

	comp_time = job.nodes['Source']['total_workload'] / network.nodes[target_node]['PS']
	
	job_completion_time = comp_time + trans_time
	job_throughput = round(1/max(comp_time, trans_time), 2)

	return target_node, job_completion_time, job_throughput


def task_allocation_no_routing_bandwidth_1(job, network):
	'''
	For each task, select the most poweful node (the completion time is the shorest), do not consider the intermediate transmission time
	Just do some small modification of the first phase of the proposed solution
	Input:
	Output: where to allocate each task
	
	for each task, find the edge node with the qualified resources;
	Within those nodes, find the node where the task can run most fast;
	return the task allocation strategies 
	'''
	
	job_assigned = []
	for task in list(nx.bfs_tree(job, 0)): # schedule the task in a layered order
		qualified_nodes = {}
		for node in network.nodes():
			if job.nodes[task]['request_resource'] < network.nodes[node]['resource']:
				qualified_nodes[node] = network.nodes[node]['PS']

		if not qualified_nodes:
			# print("this job fail to be scheduled")
			return False

		target_node = max(qualified_nodes, key=qualified_nodes.get)

		job_assigned.append(target_node)
		# update the node avail resource
		network.nodes[target_node]['resource'] = network.nodes[target_node]['resource'] - job.nodes[task]['request_resource']

	# print(job_assigned)
	# calculate the job completion time
	job_completion_time, job_throughput = cal_JCT_throughput(job, network, job_assigned)
	

	return job_assigned, job_completion_time, job_throughput


def cal_JCT_throughput(job, network, job_assigned):
	'''
	check all the data flow and their routing path (when calculate the routing path, follows the shorest path rule)
	find the overlap links,
	average the bandwidth of overlap links to all flows on that link,
	calculate the bandwidth of each flow,
	accordingly, calculate the JCT and throughput
	'''
	# find the flows and the routing path of each flow
	flows = []
	all_routing_paths = []
	for i, task in enumerate(list(nx.bfs_tree(job, 0))):
		dest_node = job_assigned[i]
		predecessors = list(job.predecessors(task))

		for predecessor in predecessors:
			index = predecessor
			if index == 'Source':
				source_node = job.nodes[index]['source']
			else:
				source_node = job_assigned[index]

			if dest_node != source_node:
				datasize = job.edges[predecessor, task]['weight']
				shorest_path_bandwidth, shorest_path = check_shorest_path(network, source_node, dest_node)
				all_routing_paths.append(shorest_path)
				flow = (source_node, dest_node, datasize, shorest_path, predecessor, task)
				# flow = (source_node, dest_node, datasize, shorest_path, source_task, dest_task)
				flows.append(flow)
				
	# print(flows)
	# calculate the transmission time of each flow
	routing_bandwidth_solutions = []
	trans_times = []
	for i in range(len(all_routing_paths)):
		flow = []
		bandwidths = []
		for edge in all_routing_paths[i]:
			counter = 0 # how many flow go through this edge/link
			for j in range(len(all_routing_paths)):
				if edge in all_routing_paths[j] or (edge[1], edge[0]) in all_routing_paths[j]:
					counter = counter + 1
			bandwidth = network.edges[edge[0], edge[1]]['weight'] / counter
			bandwidths.append(bandwidth)

		min_bandwidth = min(bandwidths)

		trans_time = flows[i][2] / min_bandwidth
		trans_times.append(trans_time)

		for edge in all_routing_paths[i]:
			edge_temp = (edge, min_bandwidth)
			flow.append(edge_temp)

		routing_bandwidth_solutions.append(flow)

	sum_comp_time = 0
	for i, node in enumerate(job_assigned):
		temp_comp_time = job.nodes[i]['CL'] / network.nodes[node]['PS']
		sum_comp_time = sum_comp_time + temp_comp_time

	job_completion_time = sum(trans_times) + sum_comp_time

	# calculate the computation time of each task  其实很简单，移去job图中是flow的边即可
	# find the distinguished edge nodes in job_assigned
	# nodes_allocated = list(set(job_assigned))
	# for node in nodes_allocated:
	# 	tasks = [index for (index, value) in enumerate(job_assigned) if value == node]
	# 	if len(tasks) > 1:
	# 		for task in tasks
	# 		temp_comp_time = job.nodes[tasks[0]]['CL'] / network.nodes[node]['PS']
	# 	else:
	job_copy = job.to_undirected()
	for flow in flows:
		job_copy.remove_edge(flow[4], flow[5])

	job_copy.remove_node('Source')

	components_time = []
	for subgraph in nx.connected_components(job_copy):
		nodeSet = job_copy.subgraph(subgraph).nodes()
		sum_comp_time = 0
		for task in nodeSet:
			temp_comp_time = job.nodes[task]['CL'] / network.nodes[job_assigned[task]]['PS']
			sum_comp_time = sum_comp_time + temp_comp_time
		components_time.append(sum_comp_time)

	if trans_times:
		max_trans_time = max(trans_times)
	else:
		max_trans_time = 0
	max_comp_time = max(components_time)

	job_throughput = round(1/max(max_comp_time, max_trans_time), 2)


	return job_completion_time, job_throughput


def task_allocation_no_routing_bandwidth_2(job, network):
	'''
	Just use the first phase of the proposed solution

	for each task, find the edge node with the qualified resources;
	Within those nodes, find the node where the task can run most fast;
	return the task allocation strategies 
	'''
	job_assigned = []
	for task in list(nx.bfs_tree(job, 0)): # schedule the task in a layered order
		# if task == 0:
		# 	target_node = job.nodes['Source']['source']
		# 	network.nodes[target_node]['resource'] = network.nodes[target_node]['resource'] - job.nodes[task]['request_resource']
		# 	job_assigned.append(target_node)
		# 	continue
		
		trans_and_comp_dict = {}
		for node in network.nodes():
			# select those nodes with qualified resources first
			if job.nodes[task]['request_resource'] >= network.nodes[node]['resource']:
				continue
			trans_and_comp = maximum_trans_and_comp_time(job, task, network, node, job_assigned)
			trans_and_comp_dict[node] = trans_and_comp

		if trans_and_comp_dict:
			target_node = min(trans_and_comp_dict, key=trans_and_comp_dict.get)
			#update the processing power of edge node
			network.nodes[target_node]['resource'] = network.nodes[target_node]['resource'] - job.nodes[task]['request_resource']

			job_assigned.append(target_node)

		else:
			# print("this job fail to be scheduled")
			return False   # this job cannot be scheduled to the edge nodes

	print(job_assigned)
	# calculate the job completion time
	job_completion_time, job_throughput = cal_JCT_throughput(job, network, job_assigned)

	return job_assigned, job_completion_time, job_throughput



def test():
	links = [('A', 'B', 10), ('B', 'F', 20), ('F', 'D', 50), ('D', 'C', 15), ('C', 'A', 5), ('B', 'E', 7), ('E', 'D', 4)]
	G = nx.Graph()
	G.add_weighted_edges_from(links)

	# add computation capability
	G.nodes['A']['PS'] = 100
	G.nodes['B']['PS'] = 50
	G.nodes['C']['PS'] = 200
	G.nodes['D']['PS'] = 150
	G.nodes['E']['PS'] = 30
	G.nodes['F']['PS'] = 100

	# add current resource
	G.nodes['A']['resource'] = 8
	G.nodes['B']['resource'] = 4
	G.nodes['C']['resource'] = 2
	G.nodes['D']['resource'] = 2
	G.nodes['E']['resource'] = 6
	G.nodes['F']['resource'] = 12

	# add max_resource
	G.nodes['A']['max_resource'] = 16
	G.nodes['B']['max_resource'] = 8
	G.nodes['C']['max_resource'] = 4
	G.nodes['D']['max_resource'] = 8
	G.nodes['E']['max_resource'] = 12
	G.nodes['F']['max_resource'] = 20


	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(G, 'weight')
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	# plt.show()

	# generate application
	
	app_links = [('Source', 0, 150), (0, 1, 50), (0, 2, 100), (2, 3, 30), (3, 4, 10)]
	app = nx.DiGraph()
	app.add_weighted_edges_from(app_links)

	app.nodes['Source']['source'] = 'A'
	app.nodes['Source']['total_request_resource'] = 8
	app.nodes['Source']['total_workload'] = 590
	app.nodes['Source']['Source_datasize'] = 150
	# add computation workload
	app.nodes[0]['CL'] = 50
	app.nodes[1]['CL'] = 150
	app.nodes[2]['CL'] = 200
	app.nodes[3]['CL'] = 90
	app.nodes[4]['CL'] = 100

	# add request_resource
	app.nodes[0]['request_resource'] = 2
	app.nodes[1]['request_resource'] = 4
	app.nodes[2]['request_resource'] = 6
	app.nodes[3]['request_resource'] = 3
	app.nodes[4]['request_resource'] = 3

	pos = nx.spring_layout(app)
	nx.draw(app, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(app, 'weight')
	nx.draw_networkx_edge_labels(app, pos, edge_labels = edge_labels)

	# plt.show()

	return G, app

if __name__ == '__main__':
	# baseline 1
	network, job = test()
	results = least_request_priority(job, network)
	if results:
		print("Baseline 1:")
		print("Target Node: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
	# baseline 2
	network, job = test()
	results = balanced_resource_allocation(job, network)
	if results:
		print("Baseline 2:")
		print("Target Node: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
	# baseline 3
	network, job = test()
	results = task_allocation_no_routing_bandwidth_1(job, network)
	if results:
		print("Baseline 3:")
		print("Job Assigned: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])
	# baseline 4
	network, job = test()
	results = task_allocation_no_routing_bandwidth_2(job, network)
	if results:
		print("Baseline 4:")
		print("Job Assigned: ", results[0])
		print("Job Completion Time: ", results[1])
		print("Throughput: ", results[2])