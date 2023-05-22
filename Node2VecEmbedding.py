import numpy as np
import scipy
from gensim.models import KeyedVectors
from networkx import diameter
from node2vec import Node2Vec
import networkx as nx
from matplotlib import pyplot as plt

#QUELLE: https://github.com/eliorc/node2vec

#Einbettung mit node2vec berechnen
def compute_node_embedding(graph, dim, l_walks, n_walks, param_p, param_q, window_size):

    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    # graph: der Graph
    # dimensions: Anzahl der Dimensionen der Einbettung am Ende
    # walk_length: Länge der random walks
    # num_walks: Anzahl der walks von jedem Knoten aus
    # p: ist per default 1, 1/p ist die Wahrscheinlichkeit zum Vorgängerknoten zurückzugehen
    # q: ist per default 1, 1/q ist die Wahrscheinlichkeit zu Knoten ohne Kante zum Vorgängerknoten zu gehen
    # falls q>1 werden nahe Knoten bevorzugt, falls q<1 ähnlicher zu DFS
    # workers: Anzahl an paralleln Abläufen von random walks
    # temp_folder: path zu einem Ordner, der bei großen Graphen genutzt werden kann
    node2vec = Node2Vec(graph, dimensions=dim, walk_length=l_walks, num_walks=n_walks, p=param_p, q=param_q, workers=1)

    # Embed nodes
    #PARAMETER: https://radimrehurek.com/gensim/models/word2vec.html
    #window: maximale Distanz zwischen dem Aktuellen und dem vorhergesagten Word in einem Satz
    #min_count: alle Wörter, die seltener vorkommen werden ignoriert
    #sg: 1 für skip-gram ansonsten cbow
    #negative: soll negative sampling verwendet werden?
    #batch_words: Anzahl an Wörter, die ein worker auf einmal übergeben bekommt
    #dimensions and workers are automatically passed (from the Node2Vec constructor)
    model = node2vec.fit(window=window_size, min_count=1, batch_words=1, sg=1, negative=1)

    #AUSGABE:
    return model

#Knoteneinbettung normalisieren
def normalize_emb(model):

    kv_norm = model.wv.get_normed_vectors()

    return kv_norm

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

#Einbettung des Graphen erstellen und speichern
def make_embedding(G_liste, dim_liste, l_walks_liste, n_walks_liste, p_liste, q_liste, w_size_liste):
    filename_list = []
    for i, tup in enumerate(G_liste):
        model = compute_node_embedding(tup[0], dim_liste[i], l_walks_liste[i], n_walks_liste[i], p_liste[i], q_liste[i], w_size_liste[i])

        emb = model.wv.__getitem__(np.arange(len(model.wv)))
        emb_norm = normalize_emb(model)

        filename = str(tup[1])+"_"+str(tup[0].number_of_nodes())+"_"+str(dim_liste[i])+"_"+str(l_walks_liste[i])+"_"+str(n_walks_liste[i])+"_"+str(p_liste[i])+"_"+str(q_liste[i])+"_"+str(w_size_liste[i])
        filename_list.append(filename)
        #Normalisierte und nicht normalisierte Einbettung speichern
        save_emb(filename, emb, normalized=False)
        save_emb(filename, emb_norm, normalized=True)

    return filename_list
