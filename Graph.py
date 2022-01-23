'''
generate the edge network
'''

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

def gen_network(noD, var, B_avg, PS_avg):
	'''
	noD: the number of edge devices
	var: variance to generate the bandwidth
	B: the average bandwidth used to generate the bandwidth of each edge
	'''

	np.random.seed(0)

	A = np.zeros((noD, noD))
	L = noD
	thres = 2*noD/5

	x_loc = np.random.rand(noD)*L
	y_loc = np.random.rand(noD)*L
	for i in range(noD):
		for j in range(noD):
			distance = math.sqrt((x_loc[i] - x_loc[j]).item()**2 + (y_loc[i] - y_loc[j]).item()**2)
			if distance <= thres and i != j:
				A[i,j] = round(B_avg + (var*B_avg)*np.random.randn(), 2)
			else:
				A[i,j] = 0

	for i in range(noD):
		for j in range(noD):
			A[i,j] = A[j,i]

	
	labels = [ i+1 for i in range(len(A)) ]
	# print(labels)
	G = nx.Graph()
	G.add_nodes_from(labels)

	for i in range(len(A)):
		for j in range(len(A)):
			if A[i,j] > 0:
				G.add_edge(i+1, j+1, weight=A[i,j])

	# add the computation power of each edge node
	PS = []
	for i in range(noD):
		ps_tmp = round(PS_avg + (var*PS_avg)*np.random.randn(), 2)
		PS.append(ps_tmp)

	for i, node in enumerate(G.nodes()):
		G.nodes[node]['PS'] = PS[i]


	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(G, 'weight')
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	plt.show()


	return G, A

if __name__ == '__main__':
	G, A = gen_network(20, 0.2, 200, 500)