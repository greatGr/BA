# from matplotlib import pyplot as plt
#
# data = [(128, 92), (96, 91), (64, 85), (48, 87), (32, 83), (16, 80.5)]
#
# # Extract x and y coordinates from the data
# x_values = [point[0] for point in data]
# y_values = [point[1] for point in data]
#
# fig = plt.figure()
#
# ax0 = fig.add_subplot(121)
#
# ax0.plot(x_values,y_values, 'bo', ms=2)
# ax0.set_xlabel("Dimension n2v Einbettung")
# ax0.set_ylabel("Test Accuracy")
#
# path = "Abbildungen/Einfluss_Dimension.png"
# fig.savefig(path)


import networkx as nx

# Create a networkx graph
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])

# Add edges to the graph
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E'), ('D', 'E'), ('D', 'F'), ('E', 'F')])

# Calculate the shortest path length between nodes 'A' and 'F'
shortest_path_length = nx.shortest_path_length(G, source='A', target='B')
shortest_path_length_2 = (len(nx.bidirectional_shortest_path(G, source='A', target='B')) - 1)

# Print the shortest path length
print("Shortest path length:", shortest_path_length, shortest_path_length_2)


