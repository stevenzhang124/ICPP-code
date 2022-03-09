import networkx as nx
import matplotlib.pyplot as plt
import random

def gen_application(noD):
	# generate application
	
	app_links = [('Source', 0, 6), (0, 1, 6), 
				(1, 2, 0.2), (1, 3, 0.2), (1, 4, 0.2), (1, 5, 0.2), (1, 6, 0.2), 
				(2, 7, 0.01), (3, 7, 0.01), (4, 7, 0.01), (5, 8, 0.01), (6, 8, 0.01), 
				(7, 9, 0.05), (8, 9, 0.05)]
	
	app = nx.DiGraph()
	app.add_weighted_edges_from(app_links)

	app.nodes['Source']['source'] = random.randint(0, noD-1)
	# app.nodes['Source']['source'] = source
	app.nodes['Source']['total_request_resource'] = 24
	app.nodes['Source']['total_workload'] = 8000
	app.nodes['Source']['Source_datasize'] = 6
	
	print("The source node is ", app.nodes['Source']['source'])

	# add computation workload
	app.nodes[0]['CL'] = 5
	app.nodes[1]['CL'] = 300
	app.nodes[2]['CL'] = 600
	app.nodes[3]['CL'] = 650
	app.nodes[4]['CL'] = 550
	app.nodes[5]['CL'] = 550
	app.nodes[6]['CL'] = 600
	app.nodes[7]['CL'] = 1500
	app.nodes[8]['CL'] = 1500
	app.nodes[9]['CL'] = 50

	# add request_resource
	app.nodes[0]['request_resource'] = 0.2
	app.nodes[1]['request_resource'] = 1
	app.nodes[2]['request_resource'] = 2
	app.nodes[3]['request_resource'] = 2
	app.nodes[4]['request_resource'] = 4
	app.nodes[5]['request_resource'] = 3
	app.nodes[6]['request_resource'] = 3
	app.nodes[7]['request_resource'] = 4
	app.nodes[8]['request_resource'] = 4
	app.nodes[9]['request_resource'] = 0.5

	# pos = nx.spring_layout(app)
	# nx.draw(app, pos, with_labels=True)

	# edge_labels = nx.get_edge_attributes(app, 'weight')
	# nx.draw_networkx_edge_labels(app, pos, edge_labels = edge_labels)

	# plt.show()

	return app

if __name__ == '__main__':
	gen_application(noD)