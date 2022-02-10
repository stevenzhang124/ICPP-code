import math
import random
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

random.seed(0)
np.random.seed(42)

class Task:
	def __init__(self,tid):
		self.tid = tid
		self.childs = []
		self.parents = []
		self.load = 0

def printWorkflow(wf,h,w):
	for i in range(h):
		print("--- LVL",(i+1),"-----")
		for j in range(w):
			if wf[i][j] != -1:
				print("[ tid = ", wf[i][j].tid, " load = ", wf[i][j].load,"MIPS childs = {",','.join(map(str,wf[i][j].childs)),"}, \
					parents = {",','.join(map(str,wf[i][j].parents)),"} ]")

def generateWorflowMatrix(wf, h, w, n, dep_data, CL_avg, var):
	mapTask = {}
	
	mat2d = [[0 for x in range(n)] for y in range(n)]
	mat1d = [[0, 0] for x in range(n)]
	tid = 0;
	for i in range(h):
		for j in range(w):
			if wf[i][j] != -1:
				mat1d[tid] = wf[i][j].load
				mapTask[wf[i][j].tid] = tid
				tid = tid  +1
	for i in range(h):
		for j in range(w):
			if wf[i][j] != -1:
				pid = wf[i][j].tid
				pind = mapTask[pid]
				for k in range(0,len(wf[i][j].childs)):
					cid = wf[i][j].childs[k][0]
					cind = mapTask[cid]
					mat2d[pind][cind] = wf[i][j].childs[k][1]
	# print(mat1d)
	# print(mapTask)
	# print(tid)
	# # print(mat2d)
	# for row in mat2d:
	# 	print(row)

	adj_matrix = mat2d
	labels = [ i for i in range(len(adj_matrix)) ]
	print(labels)
	G = nx.DiGraph()
	G.add_nodes_from(labels)

	for i in range(len(adj_matrix)):
		for j in range(len(adj_matrix)):
			if adj_matrix[i][j] > 0:
				# G.add_edge(i, j, weight=adj_matrix[i][j])
				G.add_edge(i, j)

	dep_data_list = []
	for i in range(len(list(G.edges()))):
		dep_data = round(dep_data + (var*dep_data)*np.random.randn(), 2)
		dep_data_list.append(dep_data)

	weighted_edges = []
	for i, item in enumerate(list(G.edges())):
		G.edges[item[0], item[1]]['weight'] = dep_data_list[i]
		

	# init computation load for each subtask
	CL = []
	for i in range(n):
		comp_load = round(CL_avg + (var*CL_avg)*np.random.randn(), 2)
		CL.append(comp_load)

	for i, node in enumerate(G.nodes()):
		#print(i, node)
		G.nodes[node]['CL'] = CL[i]

	# init request resource of all sub_task  [2, 16]  40% 30% 20% 10%
	task_16 = math.ceil(n * 0.1)
	task_8 = math.ceil(n * 0.2)
	task_4 = math.ceil(n * 0.3)
	task_2 = n - task_16 - task_8 - task_4

	list_tasks = list(G.nodes())
	# print(type(list_tasks))
	for i in range(len(list_tasks)):
		if i < task_16:
			task_id = random.choice(list_tasks)
			list_tasks.remove(task_id)
			G.nodes[task_id]['request_resource'] = 16
		elif i < task_16 + task_8:
			task_id = random.choice(list_tasks)
			list_tasks.remove(task_id)
			G.nodes[task_id]['request_resource'] = 8
		elif i< task_16 + task_8 + task_4:
			task_id = random.choice(list_tasks)
			list_tasks.remove(task_id)
			G.nodes[task_id]['request_resource'] = 4
		else:
			for task in list_tasks: 
					G.nodes[task]['request_resource'] = 2

	# add source node, it is a virtual node storing some meta information of the task
	G.add_edge('Source', 0)
	G.edges['Source', 0]['weight'] = 1.5*dep_data
	G.nodes['Source']['source'] = random.randint(1, 19)  # select an edge node
	G.nodes['Source']['total_request_resource'] = 64
	G.nodes['Source']['total_workload'] = sum(dep_data_list)
	G.nodes['Source']['Source_datasize'] = 1.5*dep_data

	# check the nodes and their attributes
	for node in G.nodes():
		if node == 'Source':
			print(node, G.nodes[node]['source'], G.nodes[node]['total_workload'], G.nodes[node]['total_request_resource'], G.nodes[node]['Source_datasize'])
		else:
			print(node, G.nodes[node]['CL'], G.nodes[node]['request_resource'])


	pos = nx.spring_layout(G)
	nx.draw(G, pos, with_labels=True)

	edge_labels = nx.get_edge_attributes(G, 'weight')
	nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

	plt.show()
	# print(list(nx.bfs_tree(G, 0)))

	return G


def DAGs_generate(v=10, alpha=0.5, EdgeProb=0.5, baseMIPS=1, baseBandwidth=1, CCR=0.5, Maxload=50):
	# Input Parameter
	# if len(sys.argv) < 8:
	# 	print("Enter No of tasks (V):")
	# 	v = int(input())
	# 	lower = 1/math.sqrt(v)
	# 	upper = math.sqrt(v)
	# 	alpha = -1
	# 	while alpha <= lower or alpha >= upper:
	# 		print("Enter Shape Parameter (Alpha) [",lower,",",upper,"]:")
	# 		alpha = float(input())

	# 	EdgeProb = -1
	# 	while EdgeProb < 0 or EdgeProb > 1:
	# 		print("Enter Edge Probability [0,1]: ")
	# 		EdgeProb  = float(input())

	# 	print("Enter base computational power (MIPS):")
	# 	baseMIPS = float(input())

	# 	print("Enter base Bandwidth (MBPS):")
	# 	baseBandwidth = float(input())

	# 	print("Enter CCR:")
	# 	CCR = float(input())

	# 	print("Max Task Load:")
	# 	Maxload = int(input())
	# else:
	# 	v = int(sys.argv[1])
	# 	alpha = float(sys.argv[2])
	# 	EdgeProb = float(sys.argv[3])
	# 	baseMIPS = float(sys.argv[4])
	# 	baseBandwidth = float(sys.argv[5])
	# 	CCR = float(sys.argv[6])
	# 	Maxload = int(sys.argv[7])


	edgecount = 0

	# Height and Width Calculation

	height = int(math.ceil(math.sqrt(v)/alpha)) 
	width = int(math.ceil(math.sqrt(v)*alpha))
	#print("h = ", height, "w = ", width)

	# define workflow matrix
	workflow = [[-1 for x in range(width)] for y in range(height)]
	workflowlvlcount = [0 for x in range(height)]

	currtaskid = 0
	# phase 1 mandetory task assignment to level
	for i in range(height):
		workflow[i][0] = Task(currtaskid)
		currtaskid = currtaskid + 1
		workflowlvlcount[i] = workflowlvlcount[i] + 1

	# phase 2 random task assignment to all level
	while v-currtaskid > 0:
		randlvl = random.randint(1,height-2)
		if workflowlvlcount[randlvl] == width:
			continue
		workflow[randlvl][workflowlvlcount[randlvl]] = Task(currtaskid)
		currtaskid = currtaskid + 1
		workflowlvlcount[randlvl] = workflowlvlcount[randlvl] + 1

	# phase 3 mandetory parent connection

	for lvl in range(1,height):
			for i in range(workflowlvlcount[lvl]):
				if workflowlvlcount[lvl-1] == 1:
					workflow[lvl][i].parents.append(workflow[lvl-1][0].tid)
					workflow[lvl-1][0].childs.append(workflow[lvl][i].tid)
				else:
					rnd_parent = random.randint(0,workflowlvlcount[lvl-1]-1)
					workflow[lvl][i].parents.append(workflow[lvl-1][rnd_parent].tid)
					workflow[lvl-1][rnd_parent].childs.append(workflow[lvl][i].tid);
				edgecount = edgecount + 1

	# phase 4 mandetory child connection
	for lvl in range(0,height-1):
			for i in range(workflowlvlcount[lvl]):
				if workflowlvlcount[lvl+1] == 1 and len(workflow[lvl][i].childs) == 0:
					workflow[lvl][i].childs.append(workflow[lvl+1][0].tid)
					workflow[lvl+1][0].parents.append(workflow[lvl][i].tid)
					edgecount = edgecount + 1
				elif len(workflow[lvl][i].childs) == 0:
					rnd_child = random.randint(0,workflowlvlcount[lvl+1]-1)
					workflow[lvl][i].childs.append(workflow[lvl+1][rnd_child].tid)
					workflow[lvl+1][rnd_child].parents.append(workflow[lvl][i].tid)
					edgecount = edgecount + 1

	# phase 5 random edge placement
	for lvl in range(0,height-1):
			for i in range(workflowlvlcount[lvl]):
				for j in range(workflowlvlcount[lvl+1]):
					putEdge = random.randint(0,100)
					hasEdge = workflow[lvl+1][j].tid in workflow[lvl][i].childs
					if putEdge < EdgeProb*100 and not(hasEdge):
						workflow[lvl][i].childs.append(workflow[lvl+1][j].tid)
						workflow[lvl+1][j].parents.append(workflow[lvl][i].tid)
						edgecount = edgecount + 1

	# phase 6 Assigning computational load to all tasks
	loadsum = 0
	for lvl in range(0,height):
			for i in range(workflowlvlcount[lvl]):
				workflow[lvl][i].load = random.randint(1,Maxload)
				loadsum = loadsum + workflow[lvl][i].load

	eff_ccr = CCR * (float(edgecount)/v) * (baseBandwidth/baseMIPS)
	filesum = eff_ccr * loadsum;
	filesum = int(math.ceil(filesum))

	# phase 7 distributing load to edges
	# print("file sum = ",filesum, edgecount)
	edgecountlist = []
	for i in range(edgecount):
		edgecountlist.append( random.randint( 1, int((filesum-(edgecount-(i+1)))/2)  ) )
		filesum = filesum - edgecountlist[-1]

	# phase 8 edge cost mapping
	for lvl in range(0,height-1):
		for i in range(workflowlvlcount[lvl]):
			for e in range(len(workflow[lvl][i].childs)):
				filesizeInd = random.randint(0,len(edgecountlist)-1)
				filesize = edgecountlist[filesizeInd]
				edgecountlist.pop(filesizeInd)
				workflow[lvl][i].childs[e] = [workflow[lvl][i].childs[e], filesize]
				#find parent in next lvl and update filesize
				for j in range(workflowlvlcount[lvl+1]):
					if workflow[lvl][i].childs[e][0] == workflow[lvl+1][j].tid:
						ind = workflow[lvl+1][j].parents.index(workflow[lvl][i].tid)
						workflow[lvl+1][j].parents[ind] = [workflow[lvl][i].tid, filesize]

	return workflow, height, width, v


def gen_application(noT, CL_avg, dep_data, var):
	
	workflow, height, width, v = DAGs_generate(v=noT, alpha=0.5, EdgeProb=0.5, baseMIPS=1, baseBandwidth=1, CCR=0.5, Maxload=50)

	G = generateWorflowMatrix(workflow, height, width, v, dep_data, CL_avg, var)

	print(list(nx.bfs_tree(G, 0)))

	return G

if __name__ == '__main__':
	G = gen_application(10, 300, 6000, 0.2)
	print(list(G.edges()))
