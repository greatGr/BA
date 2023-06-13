import csv
from sys import maxsize
from collections import deque
import numpy as np
import torch
from matplotlib import pyplot as plt
import ast

import ForceEmbedding
import MyGraph
import networkx as nx
from numpy import random
import NaiveEmbedding
import Node2VecEmbedding

# Berechnet (von n vielen Knoten, bis Tiefe d) alle kürzesten Wege im Graph
def compute_paths(graph, tupel_liste, id):
	asp_dictionary = {}

	for j in range(len(tupel_liste)):
		number_bfs = tupel_liste[j][0]
		depth = tupel_liste[j][1]

		# Kopierte liste mit allen Knoten des Graphen anlegen
		nodes_copy = list(graph).copy()
		paths_list = []
		for i in range(number_bfs):
			if (len(nodes_copy) > 0):
				# Zufälligen Startknoten auswählen und anschließend entfernen
				rand_i_start = random.randint(len(nodes_copy))
				start = int(nodes_copy[rand_i_start])
				del nodes_copy[rand_i_start]

				# Findet alle Wege vom Start zu allen Knoten auf der Tiefe
				# paths = {zielknoten_1 : [alle Wege dahin aufgelistet], ziel_2:..,...}
				paths = get_paths(graph, start, depth)

				#Eine Liste mit allen Wegen bilden
				paths = list(paths.values())
				paths = [e for sl in paths for e in sl]
				paths_list += paths

		asp_dictionary[depth] = paths_list

	filename = str(id) + "_" + str(graph.number_of_nodes())
	save_dict(asp_dictionary, filename)

	return asp_dictionary

# Erstellt die Datensets und speichert sie
def make_datasets(paths, graph_list, l_train, emb_type, l_test=None, normalized=True, filename_list=None):
	if filename_list is None:
		filename_list = ["naiv"] * len(graph_list)
	for i,tupel in enumerate(graph_list):
		data_pos_train, data_neg_train = create_data(paths[i], tupel, l_train[i], emb_type, normalized, filename_list[i])
		if l_test is not None:
			data_pos_test, data_neg_test = create_data(paths[i], tupel, l_test[i], emb_type, normalized, filename_list[i])
		else:
			data_pos_test, data_neg_test = torch.empty(0), torch.empty(0)

		# Gleich viele positive wie negative, funnktioniert gerade nicht, veraltet
		# Größe des kleineren Datensatzes
		#size_train = min(data_pos_tens_train.size()[0], data_neg_tens_train.size()[0])
		#size_test = min(data_pos_tens_test.size()[0], data_neg_tens_test.size()[0])
		# Tensoren zusammenfügen
		# data = torch.cat([data_pos_tens[:size], data_neg_tens[:size]], 0)

		# Alle Daten verwenden
		data_train = torch.cat([data_pos_train, data_neg_train], 0)
		data_test = torch.cat([data_pos_test, data_neg_test], 0)


		# Wie viele positive und negative samples gibt es
		print("Train Dataset: {} positive samples, {} negative samples".format(data_pos_train.size(0), data_neg_train.size(0)))
		print("Test Dataset: {} positive samples, {} negative samples".format(data_pos_test.size(0), data_neg_test.size(0)))

		if filename_list[i] == "naiv":
			filename_list[i] = str(tupel[1]) + "_" + str(tupel[0].number_of_nodes()) + "_" + "naiv"
		save_data(data_train, "Train/" + filename_list[i])
		if l_test != None:
			save_data(data_test, "Test/" + filename_list[i])

#Erstellt Tensor mit positiven und negativen Datenpaaren
def create_data(raw_paths, tupel_graph, l_liste, emb_type, normaliz, filename="Fehler"):

	if emb_type == "naiv":
		filename_naiv = str(tupel_graph[1]) + "_" + str(tupel_graph[0].number_of_nodes())
		node_emb = NaiveEmbedding.load_node_emb(filename_naiv, normalized=normaliz)
	elif emb_type == "n2v":
		filename_n2v = filename
		node_emb = Node2VecEmbedding.load_node_emb(filename_n2v, normalized=normaliz)
	else:
		print("Fehler beim Laden der Einbettung")
		return

	#csv.field_size_limit(50 * 1024 * 1024)
	#raw_paths = load_dict(str(tupel_graph[1])+"_"+str(tupel_graph[0].number_of_nodes()))

	paths_2_use = {}
	for key, value in raw_paths.items():
		if int(key) in l_liste:
			paths_2_use[key] = value

	# Datenpaare mit positivem Label erstellen
	data_pos = add_data_pos(paths_2_use, node_emb)
	# Datenpaare mit negativem Labal erstellen
	data_neg = add_data_neg(tupel_graph[0], paths_2_use, node_emb)

	#Beide Listen in numpy Arrays umwandeln
	data_pos = np.array(data_pos)
	data_neg = np.array(data_neg)

	# numpy Arrays zu Tensoren umwandeln und einmal shuffeln
	data_pos_tens_train = torch.tensor(data_pos, dtype=torch.float32)
	data_pos_tens_train = data_pos_tens_train[torch.randperm(data_pos_tens_train.size()[0])]
	data_neg_tens_train = torch.tensor(data_neg, dtype=torch.float32)
	data_neg_tens_train = data_neg_tens_train[torch.randperm(data_neg_tens_train.size()[0])]

	#AUSGABE: numpy Arrays mit den positiven und negativen Datenpaaren
	return data_pos_tens_train, data_neg_tens_train

#Findet alle Wege vom Startknoten zu einem Knoten auf einer bestimmten Tiefe
def get_paths(G, start, tiefe):

	#Adjazenzlisten dict des Graphen erstellen
	adj = nx.to_dict_of_dicts(G)
	# Knotenanzahl des Graphs
	n = len(adj)

	# Breitensuche bis bestimmte Tiefe von Startknoten
	# Liste mit Indizes aller Knoten auf diesem Level
	i_nodes = limited_bfs(adj, start, tiefe)
	empt = [[] for _ in range(len(i_nodes))]
	paths = dict(zip(i_nodes, empt))

	#Für alle Knoten auf dem Level die Breitensuche wieder zurück machen
	while (len(i_nodes) > 0):

		i = random.randint(len(i_nodes))
		new_start = i_nodes[i]
		del i_nodes[i]

		ziel = start

		# Liste, in der
		parent = [[] for _ in range(n)]

		#Führt Breitensuche durch und baut parent liste auf
		bfs_back(adj, parent, new_start)

		path = []
		# Rekursive Funktion um die Wege zu finden
		find_paths(paths, path, parent, n, ziel)

	#AUSGABE: Liste mit Listen mit allen Wegen vom Start zum Zielknoten
	return(paths)

# Breitensuche bis zu einer bestimmten Tiefe von Startknoten aus
def limited_bfs(adj, start, level):
	#Anzahl der Knoten
	n = len(adj)
	# Liste der Elternknoten initialisieren
	parent = [[] for _ in range(n)]
	# dist enthält die kürzeste Distanz vom Startknoten zu allen anderen
	dist = [maxsize for _ in range(n)]
	q = deque()

	# Insert source vertex in queue and make
	# its parent -1 and distance 0
	q.append(start)
	parent[start] = [-1]
	dist[start] = 0

	while q:
		u = q[0]
		q.popleft()
		for v in adj[u]:
			# Es wird nur bis zu einem bestimmten Level gesucht
			if (dist[v] > dist[u] + 1) and (dist[u] + 1 <= level):
				# A shorter distance is found
				# So erase all the previous parents
				# and insert new parent u in parent[v]
				dist[v] = dist[u] + 1
				q.append(v)
				parent[v].clear()
				parent[v].append(u)

			elif (dist[v] == dist[u] + 1):

				# Another candidate parent for
				# shortes path found
				parent[v].append(u)


	# Liste aller Knoten mit der gewünschten Entfernung vom Startknoten aus
	nodes = list(np.where(np.array(dist) == level)[0])
	#AUSGABE: Liste mit den Indizes alles Knoten, die die gewünschte Entfernung zum Startknoten haben
	return nodes

# Breitensuche vom Startknoten aus , parents werden gespeichert für den ganzen Graph
def bfs_back(adj,parent,start):
	#Anzahl der Knoten des Graphs
	n = len(adj)
	# Dist soll die Distanz vom Start zu jedem Knoten enthalten
	dist = [maxsize for _ in range(n)]
	q = deque()

	# Startknoten der Queue hinzufügen und parent auf -1 und dist auf 0 setzen
	q.append(start)
	parent[start] = [-1]
	dist[start] = 0

	# Bis q leer ist
	while q:
		u = q[0]
		q.popleft()
		for v in adj[u]:
			if (dist[v] > dist[u] + 1):

				# Kürzerer Weg gefunden, alle vorherigen parents löschen und neu setzen
				dist[v] = dist[u] + 1
				q.append(v)
				parent[v].clear()
				parent[v].append(u)

			elif (dist[v] == dist[u] + 1):

				# Zusätzlicher kürzester Weg und parent
				parent[v].append(u)

# Rekursive Funktion, die alle Wege vom Start zum Ziel findet und in paths speichert
#n Anzahl aller Knoten im Graph
#u Zielknoten der Suche
def find_paths(paths, path, parent,n,u):

	# Base Case
	if (u == -1):
		paths[path.copy()[-1]] = paths[path.copy()[-1]] + [path.copy()]
		return

	# Alle parents des betrachteten KNotens durchgehen
	for par in parent[u]:
		# Aktuellen Knoten in Weg einfügen
		path.append(u)
		# Rekursiver Aufruf auf seinen parents
		find_paths(paths, path, parent, n, par)

		# Aktuellen Knoten entfernen
		path.pop()

#Erstellt Datenpaare mit positivem Label zu Wegen von einem Startknoten aus
def add_data_pos(paths, node_emb):
	data_pos = []
	#print("paths ziel erster, start letzter:", paths)
	#Dictionary in Liste umwandeln

	paths = list(paths.values())
	# for i in range(len(paths)):
	# 	paths[i] = ast.literal_eval(paths[i])
	paths = [inner for outer in paths for inner in outer]

	# Dimension der Knoteneinbettung
	dim = len(node_emb.transpose())
	# Label Vektor für positive Datenpaare
	label = [1] * dim

	#Erstelle Kopie der Liste mit den Wegen
	paths_copy = paths.copy()
	#Solange die Liste nicht leer ist, wähle einen Weg aus
	while (len(paths_copy) > 0):
		i_path = random.randint(len(paths_copy))
		path = paths_copy[i_path]

		#Verwendete Kanten als verwendet markieren
		#counting_edges(graph, path)

		#Lösche ausgewählten Weg aus der Liste
		del paths_copy[i_path]

		i_start = list(node_emb[path[0]])
		i_nachbar = list(node_emb[path[1]])
		i_ziel = list(node_emb[path[-1]])

		data = [i_start, i_nachbar, i_ziel, label]
		#print("Datentripel", data)
		#Data an Liste mit Daten anhängen
		data_pos.append(data)

	return data_pos

# Erstellt Datenpaare mit negativem Label zu einem Start und Zielknoten
def add_data_neg(G, paths, node_emb):
	data_neg = []

	# Dimension der Knoteneinbettung
	dim = len(node_emb.transpose())
	# Label Vektor für positive Datenpaare
	label = [0] * dim

	for key in paths.keys():
		i_start = -1
		i_ziel = -1
		paths_key = paths[key]
		#paths_key = ast.literal_eval(paths[key])
		for i in range(len(paths_key)):
			i_start_next = paths_key[i][0]
			i_ziel_next =  paths_key[i][-1]
			if (i_start != i_start_next) or (i_ziel != i_ziel_next):
				i_start = i_start_next
				i_ziel = i_ziel_next
			else:
				continue

			#Nachbarknoten finden, der auf keinem Weg liegt
			tmp_paths = [lst for lst in paths_key if lst[0] == i_start and lst[-1] == i_ziel]
			paths_flattened = np.array(tmp_paths).flatten()
			paths_set = set(paths_flattened)
			for node in G[paths_key[i][0]]:
				if (not int(node) in paths_set):
					emb_start = list(node_emb[paths_key[i][0]])
					emb_ziel = list(node_emb[paths_key[i][-1]])
					emb_nachbar = list(node_emb[node])

					data = [emb_start, emb_nachbar, emb_ziel, label]
					# Data an Liste mit Daten anhängen
					data_neg.append(data)
	return data_neg

#Speichert Tensor mit Trainings/Testdaten in Datei
def save_data(tensor, filename):

	path = "Daten/" + filename + ".pt"

	torch.save(tensor, path)

#Lädt gespeicherten Tensor
def load_data(filename):
	path = "Daten/" + filename + ""

	data_tens = torch.load(path)

	return data_tens

def save_dict(dict, filename):
	filename_cvs = "Kürzeste Wege/" + filename + ".csv"
	with open(filename_cvs, "w", newline="") as csvfile:
		writer = csv.writer(csvfile)
		for key, value in dict.items():
			writer.writerow([key, value])

def load_dict(filename):
	path = "Kürzeste Wege/" + filename + ".csv"
	with open(path, "r") as csvfile:
		reader = csv.reader(csvfile)
		my_dict = dict(reader)

	return my_dict


#Funktioniert gerade nicht, veraltet
def use_node_emb(data_node, dim, filename):

	data_emb = []
	# Knoteneinbettung des Graphen laden
	filename = filename + "_" + str(dim)
	node_emb = MyGraph.load_node_emb("normalized/" + filename)

	# Dimension der Knoteneinbettung
	dim = len(node_emb.transpose())


	for i in range(data_node.size(dim=0)):

		# Label Vektor für negative Datenpaare
		label = [int(data_node[i][-1])] * dim

		data = [list(node_emb[int(data_node[i][0])]), list(node_emb[int(data_node[i][1])]), list(node_emb[int(data_node[i][2])]), label]
		data_emb.append(data)

	data_emb_tens = torch.tensor(data_emb, dtype=torch.float32)

	return data_emb_tens

#Plotte Graph G, Kanten unterschiedlich eingefärbt, funktioniert wahrscheinlich nicht mehr
def plot_save_graph(graph):
	# Dictionary erstellen mit den Knoten als Keys und ihren Positionen als Einträgen
	x_pos = nx.get_node_attributes(graph, "x_pos")
	y_pos = nx.get_node_attributes(graph, "y_pos")
	pos = dict()
	for i in x_pos:
		pos[i] = [x_pos[i], y_pos[i]]

	edges, edges_count = zip(*nx.get_edge_attributes(graph,'count').items())

	# Knoten des Graphen malen
	plt.subplot(121)
	nx.draw(graph, pos, node_size=50, node_color="lightblue", edgelist = edges, edge_color = edges_count, edge_cmap=plt.cm.Blues)
	#nx.draw_networkx_labels(graph, pos)

	# Abbildung speichern
	path = "Abbildungen/" + str(graph.number_of_nodes())
	plt.savefig(path)

#Zählen wie oft jede Kante in Trainings/Testdaten vorkommt
def counting_edges(graph, path):
	graph.edges[str(path[0]), str(path[1])]["count"] = graph.edges[str(path[0]), str(path[1])]["count"] + 1

def make_tripel_list(graph, filename, train_list, test_list=None, ):
	create_tripel(graph, train_list, filename, train=True)
	if test_list != None:
		create_tripel(graph, test_list, filename, train=False)

#wird vielleicht noch für force embedding gebraucht?
def create_tripel(G, tupel_liste, filename, train):
	# Leere Listen für positive Datenpaare
	tripel = []

	for j in range(len(tupel_liste)):

		number_bfs = tupel_liste[j][0]
		depth = tupel_liste[j][1]

		# Kopierte liste mit allen Knoten des Graphen anlegen
		nodes_copy = list(G).copy()

		for i in range(number_bfs):

			if (len(nodes_copy) > 0):
				# Zufälligen Startknoten auswählen und anschließend entfernen
				rand_i_start = random.randint(len(nodes_copy))
				start = int(nodes_copy[rand_i_start])
				del nodes_copy[rand_i_start]

				# Findet alle Wege vom Start zu allen Knoten auf der Tiefe
				paths = get_paths(G, start, depth)

				if (paths != {}):
					paths_list = list(paths.values())
					paths_list = [e for sl in paths_list for e in sl]

					# Erstelle Kopie der Liste mit den Wegen
					paths_list_copy = paths_list.copy()
					# Solange die Liste nicht leer ist, wähle einen Weg aus
					while (len(paths_list_copy) > 0):
						i_path = random.randint(len(paths_list_copy))
						path = paths_list_copy[i_path]

						# Verwendete Kanten als verwendet markieren
						# counting_edges(graph, path)

						# Lösche ausgewählten Weg aus der Liste
						del paths_list_copy[i_path]

						i_start = int(path[0])
						i_nachbar = int(path[1])
						i_ziel = int(path[-1])

						data = [i_start, i_nachbar, i_ziel]

						# Data an Liste mit Daten anhängen
						tripel.append(data)

	if train:
		save_tripel(tripel, "Train/" + filename)
	else:
		save_tripel(tripel, "Test/" + filename)

def save_tripel(list, filename):
	path = "Daten/" + filename + ".csv"
	with open(path, "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in list:
			writer.writerow(row)

def load_tripel(filename):
	path = "Daten/" + filename + ".csv"
	with open(path, "r") as csvfile:
		csvreader = csv.reader(csvfile)
		data = []
		for row in csvreader:
			data.append([int(i) for i in row])

	return data






