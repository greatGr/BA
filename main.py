import os

import networkx as nx
import numpy as np
import shortuuid as shortuuid

import MyGraph
import MyDataset
import NaiveEmbedding
import Node2VecEmbedding
import ForceEmbedding
import MyFeedForward
import MyAlgorithm


if __name__ == "__main__":

    # GRAPHEN ERSTELLEN/LADEN

    #Liste zu erstellender Graphen [(Anzahl Graphen, Knotenanzahl), ...(...,...)]
    tupel_liste_graphs = [(1, 128)]
    G_list = MyGraph.make_graphs(tupel_liste_graphs)


    #Einzelnen Graph laden
    # G_list = []
    # identifier = "RrzcJVeC"
    # number_nodes = "256"
    # G = MyGraph.load_graph(identifier, number_nodes)
    # print("Diam", nx.diameter(G))
    # for i in range(nx.diameter(G)-1):
    #     G_list.append((G, identifier))


    #Alle Graphen mit der bestimmten Anzahl Knoten laden
    # nodes = "64"
    # G_list = MyGraph.load_all_graphs(nodes)


    # WEGE BERECHNUNG TRAININGS/TESTDATEN

    #Alle kürzesten Wege der Graphen berechnen
    asp_list = []
    for tup in G_list:
        tupel_liste_paths = []
        for i in range(2, nx.diameter(tup[0]) + 1):
            tupel_liste_paths += [(tup[0].number_of_nodes(), i)]

        paths_dict = MyDataset.compute_paths(tup[0], tupel_liste_paths, tup[1])
        asp_list.append(paths_dict)


    # EINBETTUNGEN BERECHNEN

    # Naive Einbettung
    NaiveEmbedding.compute_embedding(G_list)

    # Parameter Node2Vec Einbettung
    dim_n2v_list = [64]
    l_walks_list = [nx.diameter(G_list[0][0])]
    n_walks_list = [10]
    # p: ist per default 1, 1/p ist die Wahrscheinlichkeit zum Vorgängerknoten zurückzugehen
    param_p_list = [10]
    # q: ist per default 1, 1/q ist die Wahrscheinlichkeit zu Knoten ohne Kante zum Vorgängerknoten zu gehen
    # falls q>1 werden nahe Knoten bevorzugt, falls q<1 ähnlicher zu DFS
    param_q_list = [0.1]
    w_size_list = [1]

    # for i in range(len(G_list)):
    #     dim_n2v = round(G_list[i][0].number_of_nodes()/i)
    #     l_walks = round(nx.diameter(G_list[i][0]) * 1)
    #     n_walks = 10
    #     param_p_1 = 10
    #     param_q_1 = 0.1
    #     window_size_1 = 1

    filename_list_n2v = Node2VecEmbedding.make_embedding(G_list, dim_n2v_list, l_walks_list, n_walks_list, param_p_list, param_q_list, w_size_list)

    #Parameter Force Embedding noch nicht benutzungsbereit
    # dim_force = 80
    # c_0 = 0.05
    # c_1 = 0.05
    # c_2 = 0.1
    # c_3 = 0.1
    #sollten kleiner als 0.5 sein
    # c_4 = 0
    # c_5 = 0
    # const_conv = 0.01
    # tolerance = 1e-2
    # filename_force = filename_graph + "_" + str(dim_force) + "_" + str(c_0) + "_" + str(c_1) +"_"+ str(c_2) + "_" + str(c_3) +"_"+ str(c_4) + "_" + str(c_5) +"_force"
    # MyDataset.make_tripel_list(G, l_train, l_test, filename_force)
    # ForceEmbedding.compute_force_embedding(G, dim_force, c_0, c_1, c_2, c_3, c_4, c_5, const_conv, tolerance, filename_force)


    # DATENSETS ERSTELLEN
    # Listen die enthalten welche Längen von Wegen für Training/Test verwendet werden sollen
    l_train = []
    for tup in G_list:
        l_train.append(tuple(range(2, nx.diameter(tup[0])+1)))
    l_test = []

    MyDataset.make_datasets(asp_list, G_list, l_train, emb_type="naiv")
    MyDataset.make_datasets(asp_list, G_list, l_train, emb_type="n2v", filename_list = filename_list_n2v)
    # MyDataset.make_datasets(G_list, emb_type="force"l_train, , )


    # TRAINING DES NEURONALEN NETZES

    #Parameter für Training neuronales Netz
    data_split = 0.7
    dim_emb = 64
    list_hidden = [64, 32, 16, 4]
    learning_rate = 0.05
    num_epochs_naiv = 3
    num_epochs_n2v = 3
    #num_epochs_force = 100

    for tup in G_list:
        filename_data_naiv = str(tup[1]) + "_" + str(tup[0].number_of_nodes()) + "_naiv.pt"
        MyFeedForward.train_classifier(filename_data_naiv, data_split, tup[0].number_of_nodes(), list_hidden, learning_rate, num_epochs_naiv)
    for i in filename_list_n2v:
        MyFeedForward.train_classifier(i + ".pt", data_split, dim_emb, list_hidden, learning_rate, num_epochs_n2v)

    # path_dir = "Daten/Train/256"
    # for filename_n2v in os.listdir(path_dir):
    #     MyFeedForward.train_classifier(filename_n2v, data_split, dim_emb, list_hidden, learning_rate, num_epochs_n2v)

    #MyFeedForward.train_classifier(filename_force, data_split, dim_force, list_hidden, learning_rate, num_epochs_force)


    # WEGE MIT CLASSIFIER BERECHNEN

    # MyAlgorithm.compute_all_paths(G, filename_data_naiv, bool_norm=True)
    for i in filename_list_n2v:
        MyAlgorithm.compute_all_paths(i, bool_norm=True)
    # MyAlgorithm.compute_all_paths(G, filename_force, bool_norm=True)


    # STUFF PLOTTEN veraltet

    # dict_dif = MyAlgorithm.load_data("128_128_10_0.1_1_n2v")
    # MyAlgorithm.plot_diffs(dict_dif, "128_128_10_0.1_1_n2v")