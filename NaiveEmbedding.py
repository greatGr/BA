import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

#Berechnet Einbettung wobei Knotenanzahl=Dimension, enthält Distanz von betrachtetem Knoten zu jedem anderen Knoten
def compute_embedding(graph, filename, norm):
    embedding = []

    for start in graph.nodes:
        dist_list = []
        for ziel in graph.nodes:
            dist = len(nx.bidirectional_shortest_path(graph, start, ziel)) - 1
            dist_list += [dist]
        embedding += [dist_list]

    #Einbettung und normalisierte Einbettung speichern
    save_emb(filename, np.array(embedding), normalized=False)
    embedding_normalized = normalize_emb(embedding)
    save_emb(filename, embedding_normalized, normalized=True)

    if norm:
        # AUSGABE: gibt normalisierte Einbettung als numpy Array zurück
        return embedding_normalized
    else:
        # AUSGABE: gibt  nicht normalisierte Einbettung als numpy Array zurück
        return np.array(embedding)

#Knoteneinbettung normalisieren
def normalize_emb(emb):

    emb_norm = []
    for i in range(len(emb)):
        norm = np.linalg.norm(emb[i])

        emb_norm += [list(emb[i] / norm)]

    return np.array(emb_norm)

#Knoteneinbettung speichern
def save_emb(file_name, arr, normalized):

    if normalized:
        #Pfad wo die Knoteneinbettung gespeichert werden soll
        path_node_emb = "Knoteneinbettung/normalized/" + file_name
    else:
        path_node_emb = "Knoteneinbettung/not_normalized/" + file_name

    np.save(path_node_emb, arr, allow_pickle=False)

    #Pfad wo das Modell gespeichert werden soll
    #path_model ="Modell_Einbettung/" + file_name
    #Speichern des Modells
    #model.save(path_model)

#Knoteneinbettung laden
def load_node_emb(file_name, normalized):

    if normalized:
        #Pfad wo zu ladende Datei gespeichert ist
        path = "Knoteneinbettung/normalized/" + file_name + ".npy"
    else:
        path = "Knoteneinbettung/not_normalized/" + file_name + ".npy"

    node_emb = np.load(path)

    #AUSGABE: Numpy Array, das die Einbettungen der einzelnen Knoten enthält
    return node_emb

#Graph und Einbettung (nur plotten und Abbildungen speichern
def plot_graph_u_emb(graph, node_embedding, file_name):

    #Dictionary erstellen mit den Knoten als Keys und ihren Positionen als Einträgen
    x_pos = nx.get_node_attributes(graph, "x_pos")
    y_pos = nx.get_node_attributes(graph, "y_pos")
    pos = dict()
    for i in x_pos:
        pos[i] = [x_pos[i], y_pos[i]]

    #Knoten des Graphen malen
    plt.subplot(121, title="Delaunay Graph")
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos)
    #Kanten malen
    nx.draw_networkx_edges(graph, pos)

    #Knoteneinbettung als Scatterplot darstellen
    #plt.subplot(122, title="Nicht normalisiert")
    #plt.scatter(node_embedding_x[:, 0], node_embedding_x[:, 1], s=50, c="lightblue")
    #for i in range(node_embedding_x.shape[0]):
        #plt.text(x=node_embedding_x[i, 0], y=node_embedding_x[i, 1], s=str(i), fontdict=dict(color="black", size = 12))

    plt.subplot(122, title="Normalisiert")
    plt.scatter(node_embedding[:, 0], node_embedding[:, 1], s=50, c="lightblue")
    for i in range(node_embedding.shape[0]):
        plt.text(x=node_embedding[i, 0], y=node_embedding[i, 1], s=str(i), fontdict=dict(color="black", size=12))

    #Abbildung speichern
    path = "Abbildungen/" + file_name
    plt.savefig(path)
    #Abbildung anzeigen
    plt.show()


    #Graph plotten und speichern nur 2D
    #plot_graph_u_emb(G, emb_norm, emb, filename)