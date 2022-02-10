'''
this file records the learning of networkx
'''

# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# import math


# G = nx.Graph()
# G.add_node('a')
# G.add_nodes_from(['b', 'c', 'd', 'e'])

# print('the nodes in the graph', G.nodes())
# print('the number of nodes in the graph', G.number_of_nodes())

# G.remove_node('a')    #删除指定节点
# G.remove_nodes_from(['b','c'])    #删除集合中的节点


# G.add_edge('d', 'e')
# print('图中所有的边', G.edges())
# print('图中边的个数', G.number_of_edges())
# G.add_edge('e', 'd', weight=4)
# # set weight of edge

# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)

# # show the weight of network in the edges
# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)



# plt.show()


# # construct graph through the adj_matrix
# adj_matrix = [[0,2,0,0,4.5,0],[3,0,1,0,7,0],[0,1,0,10,0,0],[0,0,20.6,0,1,1],[1,1,0,1,0,0],[0,0,0,1,0,0]]

# labels = [ i+1 for i in range(len(adj_matrix)) ]
# # print(labels)
# G = nx.Graph()
# G.add_nodes_from(labels)

# for i in range(len(adj_matrix)):
# 	for j in range(len(adj_matrix)):
# 		if adj_matrix[i][j] > 0:
# 			G.add_edge(i+1, j+1, weight=adj_matrix[i][j])

# adj = nx.adjacency_matrix(G)
# print(adj)

# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True)

# edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)

# # the shorest path
# path = nx.dijkstra_path(G, source=1, target=6)
# print('the shorest path between node 0 to node 6:', path)
# distance = nx.dijkstra_path_length(G, source=1, target=6)
# print('the shorest distance between node 0 to node 6:', distance)

# plt.show()


# # construct DAG
# G = nx.DiGraph()
# G.add_node(1)
# G.add_node(2)
# G.add_nodes_from([3,4,5,6])
# G.add_edge(1,3)
# G.add_edge(2,3)
# G.add_edges_from([(3,5),(3,6),(6,7)])
# print(list(G.predecessors(3)))

# nx.draw(G,node_color = 'red', with_labels=True)
# plt.show()


# # get the path and edges from a graph
# G = nx.complete_graph(4)
# paths = nx.all_simple_paths(G, source=0, target=3)
# edges = nx.all_simple_edge_paths(G, 0, 3)
# print(list(paths))
# print(list(edges))

# links = [('E', 'B'), ('C', 'D'), ('B', 'F'), ('A', 'B'), ('B', 'A'), ('F', 'B'), ('E', 'D'), ('A', 'C'), ('D', 'F'), ('B', 'E'), ('D', 'E'), ('F', 'D')]
# G = nx.Graph()
# G.add_edges_from(links)
# print(list(G.edges()))

# 最大连通子图
# import networkx as nx
# import matplotlib.pyplot as plt

# pointList = ['A','B','C','D','E','F','G']
# linkList = [('A','B'),('B','C'),('C','D'),('E','F'),('F','G'),]


# def subgraph():
#     G = nx.Graph()
#     # 转化为图结构
#     for node in pointList:
#         G.add_node(node)

#     for link in linkList:
#         G.add_edge(link[0], link[1])

#    # 画图
#     plt.subplot(211)
#     nx.draw_networkx(G, with_labels=True)
#     color =['y','g']
#     subplot = [223,224]
#     # 打印连通子图
#     for c in nx.connected_components(G):
#        # 得到不连通的子集
#         nodeSet = G.subgraph(c).nodes()
#         print(nodeSet)
#        # 绘制子图
#         subgraph = G.subgraph(c)
#         plt.subplot(subplot[0])  # 第二整行
#         nx.draw_networkx(subgraph, with_labels=True,node_color=color[0])
#         color.pop(0)
#         subplot.pop(0)

#     plt.show()

# subgraph()


import networkx as nx  #导入networkx包，命名为nx
import matplotlib.pylab as plt #导入画图工具包，命名为plt
import math
import random
# print('Number of node in ER rangom network is 30, The probability of connection is 0.2')#ER随机网络中的节点数为30.连接概率为0.2
# NETWORK_SIZE = 20 #网络节点数为30
# p = 0.2    #连接概率为0.2
# G = nx.erdos_renyi_graph(n = NETWORK_SIZE, p = p)#用erdos_renyi_graph(n,p)方法生成一个含有n个节点、以概率p连接的ER随机图
# ps = nx.spring_layout(G)
# nx.draw(G,ps,width=0.6,node_size=10)#绘制边的宽度为0.6，节点尺寸为10的网络G图
# # plt.savefig('fig.png',bbox_inches='tight') #将图像存为一个png格式的图片文件
# plt.show() #在窗口中显示这幅图像
# # nx.write_gexf(G, 'suiji_file.gexf')#此图写成.gexf格式

G_temp = nx.random_graphs.random_regular_graph(3, 20)
# edges_list = list(G_temp.edges())
# print(edges_list)

# print(G_temp.edges(1))
num_remove = math.ceil(20*0.5)
i = 0
nodes_list = list(G_temp.nodes())
select_nodes = []
while i < num_remove:
	select_node = random.choice(nodes_list)
	select_nodes.append(select_node)
	nodes_list.remove(select_node)
	i = i + 1
print(select_nodes)

for node in select_nodes:
	edges_list = list(G_temp.edges(node))
	print(edges_list)
	temp = len(edges_list)
	print(temp)
	if temp > 1:
		G_temp.remove_edge(edges_list[-1][0], edges_list[-1][1])

edges_list = list(G_temp.edges())
print(edges_list)
print(G_temp.number_of_nodes())
nx.draw(G_temp,with_labels=True)
plt.show() 