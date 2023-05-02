import networkx as nx

import MyAlgorithm
import MyDataset
import MyFeedForward
import MyGraph
import NaiveEmbedding
import Node2VecEmbedding
from MyDataset import make_datasets
from MyGraph import new_graph
from NaiveEmbedding import compute_embedding

if __name__ == "__main__":

    #Parameter Graph
    nodes = 128

    filename_graph = str(nodes)

    # Erstellen Graph
    #G = MyGraph.new_graph(nodes, filename_graph)
    G = MyGraph.load_graph(str(nodes))
    #print("Diameter: ", nx.diameter(G))

    # Naive Einbettung der Knoten
    filename_emb_naiv = filename_graph + "_naiv"
    #embedding_naiv = NaiveEmbedding.compute_embedding(G, filename_emb_naiv, bool_norm=True)

    # Parameter Node2Vec Einbettung
    dim = nodes
    l_walks = int(nx.diameter(G) * 1)
    n_walks = 10
    # p: ist per default 1, 1/p ist die Wahrscheinlichkeit zum Vorgängerknoten zurückzugehen
    # q: ist per default 1, 1/q ist die Wahrscheinlichkeit zu Knoten ohne Kante zum Vorgängerknoten zu gehen
    # falls q>1 werden nahe Knoten bevorzugt, falls q<1 ähnlicher zu DFS
    param_p = 100
    param_q = 0.01
    window_size = 1

    filename_emb_n2v = str(nodes) +"_"+ str(dim) +"_"+ str(param_p) +"_"+ str(param_q) +"_"+ str(window_size)
    #Node2VecEmbedding.make_embedding(G, filename_emb_n2v, dim, l_walks, n_walks, param_p, param_q, window_size)

    # Trainings-und Testdaten erstellen
    filename_data_naiv = filename_emb_naiv
    filename_data_n2v = filename_emb_n2v
    filename_paths = filename_graph + "_asp"

    # Listen (Anzahl Startknoten, Länge der Wege) für erstellen der Trainings-und Testdaten
    l_train = []
    for i in range(2, nx.diameter(G)+1):
        l_train += [(nodes, i)]
    l_test = []

    #Datensets erstellen
    #MyDataset.make_datasets(G, l_train, l_test, filename_emb_naiv, filename_data_naiv, emb_type="naiv", normalized=True)
    #MyDataset.make_datasets(G, l_train, l_test, filename_emb_n2v, filename_data_n2v, emb_type="n2v", normalized=True)

    #Parameter für Training neuronales Netz
    data_split = 0.7
    dim_emb = nodes
    list_hidden = [128, 64, 8]
    learning_rate = 0.05
    num_epochs_naiv = 400
    num_epochs_n2v = 50

    #Neuronales Netz mit naiver Einbettung trainieren
    #MyFeedForward.train_classifier(filename_data_naiv, data_split, dim_emb, list_hidden, learning_rate, num_epochs_naiv)
    #Neuronales Netz mit n2v Einbettung trainieren
    #MyFeedForward.train_classifier(filename_data_n2v, data_split, dim_emb, list_hidden, learning_rate, num_epochs_n2v)


    #Alle Wege im Graph berechnen
    #MyAlgorithm.compute_all_paths(G, filename_data_naiv, bool_norm=True)
    MyAlgorithm.compute_all_paths(G, filename_data_n2v, bool_norm=True)

