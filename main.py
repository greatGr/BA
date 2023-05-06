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
    print("Diameter: ", nx.diameter(G))

    # Naive Einbettung der Knoten
    filename_emb_naiv = filename_graph + "_naiv"
    #embedding_naiv = NaiveEmbedding.compute_embedding(G, filename_emb_naiv, bool_norm=True)

    # Parameter Node2Vec Einbettung
    dim_1 = 128
    l_walks_1 = int(nx.diameter(G) * 1)

    n_walks = 10
    # p: ist per default 1, 1/p ist die Wahrscheinlichkeit zum Vorgängerknoten zurückzugehen
    # q: ist per default 1, 1/q ist die Wahrscheinlichkeit zu Knoten ohne Kante zum Vorgängerknoten zu gehen
    # falls q>1 werden nahe Knoten bevorzugt, falls q<1 ähnlicher zu DFS
    param_p_1 = 10
    param_q_1 = 0.1
    window_size_1 = 1


    filename_n2v_11 = str(nodes) +"_"+ str(dim_1) +"_"+ str(param_p_1) +"_"+ str(param_q_1) +"_"+ str(window_size_1)
    Node2VecEmbedding.make_embedding(G, filename_n2v_11, dim_1, l_walks_1, n_walks, param_p_1, param_q_1, window_size_1)




    # Trainings-und Testdaten erstellen
    #filename_paths = filename_graph + "_asp"

    # Listen (Anzahl Startknoten, Länge der Wege) für erstellen der Trainings-und Testdaten
    l_train = []
    for i in range(2, nx.diameter(G)+1):
        l_train += [(nodes, i)]
    l_test = []

    #Datensets erstellen
    #MyDataset.make_datasets(G, l_train, l_test, filename_emb_naiv, filename_data_naiv, emb_type="naiv", normalized=True)

    MyDataset.make_datasets(G, l_train, l_test, filename_n2v_11, emb_type="n2v", normalized=True)


    #Parameter für Training neuronales Netz
    data_split = 0.7
    #dim_emb =
    list_hidden = [128, 64, 16, 4]
    learning_rate = 0.05
    num_epochs_naiv = 400
    num_epochs_n2v = 5

    #Neuronales Netz mit naiver Einbettung trainieren
    #MyFeedForward.train_classifier(filename_data_naiv, data_split, dim_emb, list_hidden, learning_rate, num_epochs_naiv)

    #Neuronales Netz mit n2v Einbettung trainieren
    MyFeedForward.train_classifier(filename_n2v_11, data_split, dim_1, list_hidden, learning_rate, num_epochs_n2v)


    #Alle Wege im Graph berechnen
    #MyAlgorithm.compute_all_paths(G, filename_data_naiv, bool_norm=True)
    #MyAlgorithm.compute_all_paths(G, filename_data_n2v, bool_norm=True)

