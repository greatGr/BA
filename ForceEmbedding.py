import math

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as dist
from scipy.spatial import distance

import MyDataset


def compute_force_embedding(graph, dim, c_0, c_1, c_2, c_3, c_4, c_5, c6, c7, max_iter, const_conv, tolerance, split, filename):

    #Daten Tripel laden
    data_train = MyDataset.load_tripel("Train/"+ filename)

    #Wie viel prozent aller daten sollen verwednet werden
    data_train = np.array(data_train)
    np.random.shuffle(data_train)
    s = math.ceil(data_train.shape[0] * split)
    emb_train = data_train[:s]
    emb_train = emb_train.tolist()
    print("Anzahl daten", len(emb_train))

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

    # Max iterationene durchlaufen, außer es ist vorher schon bestimmt gut
    while max_iter > iter_count:
        print(iter_count)
        # Alte pos speichern
        old_pos = pos.copy()

        # Kräfte berechnen
        forces = calculate_forces(graph, pos, emb_train, c_0, c_1, c_2, c_3, c_4, c_5, c6, c7)

        # Neue Positionene der Knoten
        pos = pos + forces * const_conv

        #Falls nur noch sehr kleine kräfte wirken
        avr_force = sum(forces) / len(forces)
        val = np.linalg.norm(avr_force)
        if val < tolerance:
            print("Vorher beendet nach ", iter_count)
            break
        iter_count += 1

        # Aktuelle Einbettung plotten
        # if dim == 2:
        #     show_graph(graph, pos)

        # Compute the change in node positions
        delta_pos = np.linalg.norm(pos - old_pos)
    print("finaler val", val)
    mean_edge = compute_mean_edge(graph, pos)

    #Normalisierte und nicht normalisierte Einbettung speichern
    save_emb(filename, pos, normalized=False)

    #Nur in 2D, Graph mit Einbettung plotten
    if dim == 2:
        draw_graph(graph, pos, filename)

#Funktion, die alle Kräfte berechnet
def calculate_forces(G, pos, data_train, c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7):

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
        d_eucl_s_t = dist.euclidean(pos[s], pos[t])
        d_node_v_t = nx.shortest_path_length(G, source=v, target=t)
        d_node_s_t = nx.shortest_path_length(G, source=s, target=t)

        vec_sv_norm = normalize_vector(pos[v] - pos[s])
        vec_st_norm = normalize_vector(pos[t] - pos[s])
        forces[s] += vec_sv_norm * (d_eucl_s_v - 1) * c_0 + vec_st_norm * (d_eucl_s_t - d_node_s_t) * c_6
        scale[s] += 2
        vec_tv_norm = normalize_vector(pos[v] - pos[t])
        vec_ts_norm = normalize_vector(pos[s] - pos[t])
        forces[t] += vec_tv_norm * (d_eucl_v_t - d_node_v_t) * c_1 + vec_ts_norm * (d_eucl_s_t - d_node_s_t) * c_7
        scale[t] += 2
        vec_vs_norm = normalize_vector(pos[s] - pos[v])
        vec_vt_norm = normalize_vector(pos[t] - pos[v])
        forces[v] += vec_vs_norm * (d_eucl_s_v - 1) * c_2 + vec_vt_norm * (d_eucl_v_t - d_node_v_t) * c_3
        scale[v] += 2

        for n in list(G.neighbors(v)):
            d_eucl_n_v = dist.euclidean(pos[n], pos[v])
            vec_nv_norm = normalize_vector((pos[v] - pos[n]))
            forces[n] += vec_nv_norm * (d_eucl_n_v - 1) * c_4
            scale[n] += 1
            vec_vn_norm = normalize_vector((pos[n] - pos[v]))
            forces[v] += vec_vn_norm * (d_eucl_n_v - 1) * c_5
            scale[v] += 1

    #Durchschnittliche kräfte verwenden
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

#berechnet durchschnittliche kantenlänge am ende in einbettung
def compute_mean_edge(G, emb):
    length_list = []
    for source_node, target_node in G.edges():
        source_embedding = emb[int(source_node)]
        target_embedding = emb[int(target_node)]

        length = distance.euclidean(source_embedding, target_embedding)
        length_list += [length]

    mean = sum(length_list)/len(length_list)
    return mean



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

def normalize_vector(vector):
    magnitude = np.linalg.norm(vector)
    normalized_vector = vector / magnitude
    return normalized_vector



