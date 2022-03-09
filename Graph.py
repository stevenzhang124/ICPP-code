'''
generate the edge network
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def gen_network(noD, var, B_avg, PS_avg):
	'''
	noD: the number of edge devices
	var: variance to generate the bandwidth
	B: the average bandwidth used to generate the bandwidth of each edge
	'''

	#### the network generated with this method will be much complex, many routing path
	# np.random.seed(0)

	# A = np.zeros((noD, noD))
	# L = noD
	# thres = 2*noD/5

	# x_loc = np.random.rand(noD)*L
	# y_loc = np.random.rand(noD)*L
	# for i in range(noD):
	# 	for j in range(noD):
	# 		distance = math.sqrt((x_loc[i] - x_loc[j]).item()**2 + (y_loc[i] - y_loc[j]).item()**2)
	# 		if distance <= thres and i != j:
	# 			A[i,j] = round(B_avg + (var*B_avg)*np.random.randn(), 2)
	# 		else:
	# 			A[i,j] = 0

	# for i in range(noD):
	# 	for j in range(noD):
	# 		A[i,j] = A[j,i]

	
	# labels = [ i+1 for i in range(len(A)) ]
	# # print(labels)
	# G = nx.Graph()
	# G.add_nodes_from(labels)

	# for i in range(len(A)):
	# 	for j in range(len(A)):
	# 		if A[i,j] > 0:
	# 			G.add_edge(i+1, j+1, weight=A[i,j])

	#  graph generated with networkx
	np.random.seed(0)
	random.seed(0)

	G = nx.random_graphs.random_regular_graph(3, noD)
	num_remove = math.ceil(noD*0.2)
	i = 0
	nodes_list = list(G.nodes())
	select_nodes = []
	while i < num_remove:
		select_node = random.choice(nodes_list)
		select_nodes.append(select_node)
		nodes_list.remove(select_node)
		i = i + 1
	# print(select_nodes)

	for node in select_nodes:
		edges_list = list(G.edges(node))
		# print(edges_list)
		temp = len(edges_list)
		# print(temp)
		if temp > 1:
			G.remove_edge(edges_list[-1][0], edges_list[-1][1])


	# check if the generated graph is all connected
	if nx.is_connected(G):
		print("The generated network is connected")


	for i, item in enumerate(list(G.edges())):
		G.edges[item[0], item[1]]['weight'] = round(B_avg + (var*B_avg)*np.random.randn(), 2)


	# add the computation power of each edge node
	PS = []
	for i in range(noD):
		ps_tmp = round(PS_avg + (var*PS_avg)*np.random.randn(), 2)
		PS.append(ps_tmp)

	for i, node in enumerate(G.nodes()):
		G.nodes[node]['PS'] = PS[i]

	for u, v, d in G.edges(data='weight'):
		print((u,v,d))
	# add the maximum resource (memory) of the edge nodes
	# the range [2, 512], small 2 4 8, medium 16 32 64, large 128 256 512
	num_large_nodes = math.ceil(noD * 0.1)
	num_medium_nodes = math.ceil(noD * 0.3)
	num_small_nodes = noD - num_large_nodes - num_medium_nodes

	list_large_nodes = [128, 256, 512]
	list_medium_nodes = [16, 32, 64]
	list_small_nodes = [2, 4, 8]

	for i, node in enumerate(G.nodes()):
		if i < num_large_nodes:
			G.nodes[node]['max_resource'] = random.choice(list_large_nodes)
		elif i < num_large_nodes + num_medium_nodes:
			G.nodes[node]['max_resource'] = random.choice(list_medium_nodes)
		else:
			G.nodes[node]['max_resource'] = random.choice(list_small_nodes)


	# initial the available resource of the edge nodes
	for node in G.nodes():
		G.nodes[node]['resource'] = G.nodes[node]['max_resource']

	# check the nodes and their attributes
	for node in G.nodes():
		print(node, G.nodes[node]['PS'], G.nodes[node]['max_resource'], G.nodes[node]['resource'])


	# pos = nx.spring_layout(G)
	# nx.draw(G, pos, with_labels=True)

	# edge_labels = nx.get_edge_attributes(G, 'weight')
	# nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	# plt.show()


	# return G, A
	return G

if __name__ == '__main__':
	# G, A = gen_network(20, 0.2, 200, 500)
	G = gen_network(50, 0.2, 200, 500)