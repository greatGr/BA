import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.spatial.distance as dist

import MyDataset

#Funktion, die alle Kräfte berechnet
def calculate_forces(G, pos, data_train, c_0, c_1, c_2, c_3, c_4, c_5):
    #Liste mit allen wirkenden Kräften initialisieren
    forces = np.zeros((len(G.nodes()), pos.shape[1]))

    for i in range(len(data_train)):
        tripel = data_train[i]
        s = tripel[0]
        v = tripel[1]
        t = tripel[2]

        dist_eucl_s_v = dist.euclidean(pos[s], pos[v])
        dist_eucl_v_t = dist.euclidean(pos[v], pos[t])
        dist_node_s_v = (len(nx.bidirectional_shortest_path(G, str(s), str(v))) - 1)
        dist_node_v_t = (len(nx.bidirectional_shortest_path(G, str(v), str(t))) - 1)

        forces[s] += (pos[v] - pos[s]) * (dist_eucl_s_v - dist_node_s_v) * c_0
        forces[t] += (pos[v] - pos[t]) * (dist_eucl_v_t - dist_node_v_t) * c_1
        forces[v] += (pos[s] - pos[v]) * (dist_eucl_s_v - dist_node_s_v) * c_2 + (pos[t] - pos[v]) * (dist_eucl_v_t - dist_node_v_t) * c_3

        for n in list(G.neighbors(str(v))):
            dist_eucl_n_v = dist.euclidean(pos[int(n)], pos[v])

            forces[int(n)] += (pos[v] - pos[int(n)]) * (dist_eucl_n_v - 0.1) * c_4
            forces[v] += (pos[int(n)] - pos[v]) * (dist_eucl_n_v - 0.1) * c_5


        #print("forces aufegteilt", forces[s], forces[t], forces[v])

    return forces

def compute_force_embedding(graph, dim, c_0, c_1, c_2, c_3, c_4, c_5, const_conv, tolerance, filename):

    #Daten Tripel laden
    data_train = MyDataset.load_tripel("Train/"+ filename)

    #noch samples die verwendet werden solllen auswählen

    # Zufällige Anfangspositionen der Knoten
    pos = np.random.rand(len(graph.nodes()), dim)


    delta_pos = np.inf

    iter_count = 0

    # Durchlaufen bis es konvergiert
    while delta_pos > tolerance:
        iter_count += 1

        # Alte pos speichern
        old_pos = pos.copy()

        # Kräfte berechnen
        forces = calculate_forces(graph, pos, data_train, c_0, c_1, c_2, c_3, c_4, c_5)

        # Neue Positionene der Knoten
        pos += forces * const_conv
        #print("Pos", pos, "forces", forces)

        # Compute the change in node positions
        delta_pos = np.linalg.norm(pos - old_pos)

        print(iter_count, delta_pos)
        #print("pos", pos)



    print("Number iterations force emb:", iter_count)

    pos_norm = normalize_emb(pos)
    save_emb(filename, pos_norm, normalized=True)
    save_emb(filename, pos, normalized=False)

    draw_graph(graph, pos, filename)


def draw_graph(graph, pos, filename):

    pos_dict = dict()
    for i in range(len(pos)):
        pos_dict[str(i)] = []
        for j in range(len(pos[i])):
            print(pos[i][j])
            pos_dict[str(i)] += [pos[i][j]]

    print("Pos vorher", pos)
    # Knoten des Graphen malen
    plt.subplot(121, title="Force Embedding")
    nx.draw_networkx_nodes(graph, pos_dict, node_size=50, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos_dict)
    # Kanten malen
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



