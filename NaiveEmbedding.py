import numpy as np
import networkx as nx

#Berechnet Einbettung, enthält Distanz von betrachtetem Knoten zu jedem anderen Knoten
def compute_embedding(G_liste):
    for tupel in G_liste:
        embedding = []

        for start in tupel[0].nodes:
            dist_list = []
            for ziel in tupel[0].nodes:
                dist = len(nx.bidirectional_shortest_path(tupel[0], start, ziel)) - 1
                dist_list += [dist]
            embedding += [dist_list]

        embedding_normalized = normalize_emb(embedding)

        filename = str(tupel[1]) + "_" + str(tupel[0].number_of_nodes()) + "_" + "naiv"
        #Einbettung und normalisierte Einbettung speichern
        save_emb(filename, np.array(embedding), normalized=False)
        save_emb(filename, embedding_normalized, normalized=True)

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

#Knoteneinbettung laden
def load_node_emb(file_name, normalized):
    if normalized:
        #Pfad wo zu ladende Datei gespeichert ist
        path = "Knoteneinbettung/normalized/" + file_name + "_naiv.npy"
    else:
        path = "Knoteneinbettung/not_normalized/" + file_name + "_naiv.npy"

    node_emb = np.load(path)

    #AUSGABE: Numpy Array, das die Einbettungen der einzelnen Knoten enthält
    return node_emb

