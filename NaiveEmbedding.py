import numpy as np
import networkx as nx


#Berechnet Einbettung, enthält Distanz von betrachtetem Knoten zu jedem anderen Knoten
def compute_embedding(graph, filename, bool_norm):
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

    #print("Nicht normalisierte naive Einbettung:", np.array(embedding))
    #print("Normalisierte naive Einbettung:", embedding_normalized)

    if bool_norm:
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

