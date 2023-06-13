import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as dist

import MyDataset


def compute_force_embedding(graph, dim, c_0, c_1, c_2, c_3, c_4, c_5, const_conv, tolerance, filename):

    #Daten Tripel laden
    data_train = MyDataset.load_tripel("Train/"+ filename)

    #noch samples die verwendet werden solllen auswählen

    # Approx durchschnittliche Kantenlänge
    average_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    mean_edge_length = 1 / average_degree

    # Zufällige Anfangspositionen der Knoten aus dem Einheitsquadrat
    pos = np.random.rand(len(graph.nodes()), dim)
    #Größer skalieren
    pos = pos/mean_edge_length

    delta_pos = np.inf

    iter_count = 0

    # Durchlaufen bis es konvergiert
    while delta_pos > tolerance:
    #while 5 > iter_count:
        iter_count += 1

        # Alte pos speichern
        old_pos = pos.copy()

        # Kräfte berechnen
        forces = calculate_forces(graph, pos, data_train, c_0, c_1, c_2, c_3, c_4, c_5)

        # Neue Positionene der Knoten
        pos = pos + forces * const_conv

        # Aktuelle Einbettung plotten
        # if dim == 2:
        #     show_graph(graph, pos)

        # Compute the change in node positions
        delta_pos = np.linalg.norm(pos - old_pos)

        print(iter_count, "Unterschied pos", delta_pos)

    #Einbettung normalisieren
    pos_norm = normalize_emb(pos)
    #Normalisierte und nicht normalisierte Einbettung speichern
    save_emb(filename, pos_norm, normalized=True)
    save_emb(filename, pos, normalized=False)

    #Nur in 2D, Graph mit Einbettung plotten
    if dim == 2:
        draw_graph(graph, pos, filename)

#Funktion, die alle Kräfte berechnet
def calculate_forces(G, pos, data_train, c_0, c_1, c_2, c_3, c_4, c_5):

    #Liste mit allen wirkenden Kräften initialisieren
    forces = np.zeros((len(G.nodes()), pos.shape[1]))
    scale = np.zeros((len(G.nodes()), 1))

    # Alle Trainingstripel durchgehen
    for i in range(len(data_train)):
        s = data_train[i][0]
        v = data_train[i][1]
        t = data_train[i][2]

        d_eucl_s_v = dist.euclidean(pos[s], pos[v])
        d_eucl_v_t = dist.euclidean(pos[v], pos[t])
        d_node_s_v = nx.shortest_path_length(G, source=s, target=v)
        d_node_v_t = nx.shortest_path_length(G, source=v, target=t)

        forces[s] += (pos[v] - pos[s]) * (d_eucl_s_v - d_node_s_v) * c_0
        scale[s] += 1
        forces[t] += (pos[v] - pos[t]) * (d_eucl_v_t - d_node_v_t) * c_1
        scale[t] += 1
        forces[v] += (pos[s] - pos[v]) * (d_eucl_s_v - d_node_s_v) * c_2 + (pos[t] - pos[v]) * (d_eucl_v_t - d_node_v_t) * c_3
        scale[v] += 2

        for n in list(G.neighbors(v)):
            d_eucl_n_v = dist.euclidean(pos[n], pos[v])

            forces[n] += (pos[v] - pos[n]) * (d_eucl_n_v - 1) * c_4
            scale[n] += 1
            forces[v] += (pos[n] - pos[v]) * (d_eucl_n_v - 1) * c_5
            scale[v] += 1

    forces = forces / scale

    return forces

#Zeigt aktuelle EInbettung
def show_graph(graph, pos):
    pos_dict = dict()
    for i in range(len(pos)):
        pos_dict[i] = []
        for j in range(len(pos[i])):
            pos_dict[i] += [pos[i][j]]

    # Knoten des Graphen malen
    plt.subplot(121, title="Force Embedding")
    nx.draw_networkx_nodes(graph, pos_dict, node_size=50, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos_dict)
    #Kanten malen
    nx.draw_networkx_edges(graph, pos_dict)

    # Abbildung speichern
    plt.show()

#Malt und speichert finale Einbettung
def draw_graph(graph, pos, filename):
    pos_dict = dict()
    for i in range(len(pos)):
        pos_dict[i] = []
        for j in range(len(pos[i])):
            pos_dict[i] += [pos[i][j]]

    # Knoten des Graphen malen
    plt.subplot(121, title="Force Embedding")
    nx.draw_networkx_nodes(graph, pos_dict, node_size=50, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos_dict)
    #Kanten malen
    nx.draw_networkx_edges(graph, pos_dict)

    # Abbildung speichern
    path = "Abbildungen/Graph_Force_Emb/" + filename + ".png"
    plt.savefig(path)

#Knoteneinbettung und Modell speichern
def save_emb(file_name, arr, normalized):

    if normalized:
        #Pfad wo die Knoteneinbettung gespeichert werden soll
        path_node_emb = "Knoteneinbettung/normalized/" + file_name
    else:
        # Pfad wo die Knoteneinbettung gespeichert werden soll
        path_node_emb = "Knoteneinbettung/not_normalized/" + file_name

    np.save(path_node_emb, arr, allow_pickle=False)

#Knoteneinbettung laden
def load_node_emb(file_name, normalized):

    if normalized:
        #Pfad wo zu ladende Datei gespeichert ist
        path = "Knoteneinbettung/normalized/" + file_name + ".npy"
    else:
        # Pfad wo zu ladende Datei gespeichert ist
        path = "Knoteneinbettung/not_normalized/" + file_name + ".npy"

    node_emb = np.load(path)

    #AUSGABE: Numpy Array, das die Einbettungen der einzelnen Knoten enthält
    return node_emb

def normalize_emb(emb):

    emb_norm = []
    for i in range(len(emb)):
        norm = np.linalg.norm(emb[i])

        emb_norm += [list(emb[i] / norm)]

    return np.array(emb_norm)



