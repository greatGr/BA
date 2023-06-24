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
    # tupel_liste_graphs = [(1, 20)]
    # G_list = MyGraph.make_graphs(tupel_liste_graphs)


    #Einzelnen Graph laden
    G_list = []
    # identifier = "WgVnqVBE"
    # number_nodes = "20"
    # identifier = "9GExUnfS"
    # number_nodes = "5"
    identifier = "4iBPVTty"
    number_nodes = "100"
    G = MyGraph.load_graph(identifier, number_nodes)
    G_list.append((G, identifier))
    G_list.append((G, identifier))
    G_list.append((G, identifier))
    G_list.append((G, identifier))
    G_list.append((G, identifier))
    G_list.append((G, identifier))

    # print("Diam", nx.diameter(G))
    # for i in range(6):
    #G_list.append((G, identifier))

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
    #NaiveEmbedding.compute_embedding(G_list)

    # Parameter Node2Vec Einbettung
    dim_n2v_list = [50] * 6
    l_walks_list = [nx.diameter(G_list[0][0])] * 6
    n_walks_list = [5, 10, 20, 30, 50, 100]
    # p: ist per default 1, 1/p ist die Wahrscheinlichkeit zum Vorgängerknoten zurückzugehen
    param_p_list = [10] * 6
    # q: ist per default 1, 1/q ist die Wahrscheinlichkeit zu Knoten ohne Kante zum Vorgängerknoten zu gehen
    # falls q>1 werden nahe Knoten bevorzugt, falls q<1 ähnlicher zu DFS
    param_q_list = [0.1] * 6
    w_size_list = [2] * 6

    # for i in range(len(G_list)):
    #     dim_n2v = round(G_list[i][0].number_of_nodes()/i)
    #     l_walks = round(nx.diameter(G_list[i][0]) * 1)
    #     n_walks = 10
    #     param_p_1 = 10
    #     param_q_1 = 0.1
    #     window_size_1 = 1

    filename_list_n2v = Node2VecEmbedding.make_embedding(G_list, dim_n2v_list, l_walks_list, n_walks_list, param_p_list, param_q_list, w_size_list)

    #Parameter Force Embedding noch nicht benutzungsbereit
    # dim_force = 10
    # #für kraft sv und vt und st
    # c_0 = 0.5
    # c_1 = 0.5
    # c_2 = 0.5
    # c_3 = 0.5
    # c_6 = 0.5
    # c_7 = 0.5
    # #für Kraft länge kanten
    # c_4 = 0.4
    # c_5 = 0.4
    # const_conv = 0.2
    # max_iter = 600
    # tolerance = 1e-3
    # split = 0.3
    #
    # for tup in G_list:
    #     filename_force = tup[1] + "_" + str(tup[0].number_of_nodes()) + "_" + str(dim_force) + "_" + str(c_0) + "_" + str(c_1) +"_"+ str(c_2) + "_" + str(c_3) +"_"+ str(c_4) + "_" + str(c_5) +"_"+ str(c_6) + "_" + str(c_7)+"_force"
    #     #Liste aus welchen Wegen Tripel erstellt werden
    #     l_train = []
    #     for i in range(2, nx.diameter(tup[0])+1):
    #         l_train += [(tup[0].number_of_nodes(), i)]
    #
    #     MyDataset.make_tripel_list(tup[0], filename_force, l_train)
    #
    #     ForceEmbedding.compute_force_embedding(tup[0], dim_force, c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, max_iter, const_conv, tolerance, split, filename_force)


    # DATENSETS ERSTELLEN
    #Listen die enthalten welche Längen von Wegen für Training/Test verwendet werden sollen
    l_train = []
    for tup in G_list:
         l_train.append(tuple(range(2, nx.diameter(tup[0])+1)))
    l_test = []

    #MyDataset.make_datasets(asp_list, G_list, l_train, emb_type="naiv")
    MyDataset.make_datasets(asp_list, G_list, l_train, emb_type="n2v", filename_list = filename_list_n2v)
    # MyDataset.make_datasets(G_list, emb_type="force"l_train, , )


    # TRAINING DES NEURONALEN NETZES

    #Parameter für Training neuronales Netz
    data_split = 1
    dim_emb = 50
    list_hidden = [50, 20, 10, 5]
    learning_rate = 0.05
    # #num_epochs_naiv = 3
    num_epochs_n2v = 100
    #num_epochs_force = 100

    #Training node2vec
    for i in filename_list_n2v:
        MyFeedForward.train_classifier(i + ".pt", data_split, dim_emb, list_hidden, learning_rate, num_epochs_n2v)

    # path_dir = "Daten/Train/256"
    # for filename_n2v in os.listdir(path_dir):
    #     MyFeedForward.train_classifier(filename_n2v, data_split, dim_emb, list_hidden, learning_rate, num_epochs_n2v)

    #Training naive Einbettung
    # for tup in G_list:
    #      filename_data_naiv = str(tup[1]) + "_" + str(tup[0].number_of_nodes()) + "_naiv.pt"
    #      MyFeedForward.train_classifier(filename_data_naiv, data_split, tup[0].number_of_nodes(), list_hidden, learning_rate, num_epochs_naiv)


    # WEGE MIT CLASSIFIER BERECHNEN

    # MyAlgorithm.compute_all_paths(G, filename_data_naiv, bool_norm=True)
    # for i in filename_list_n2v:
    #     MyAlgorithm.compute_all_paths(i, bool_norm=True)
    #MyAlgorithm.compute_all_paths(filename_force, bool_norm=False)
    # filename_emb = "2M93qusV_128_96_10_10_10_0.1_1"
    # MyAlgorithm.compute_all_paths(filename_emb, bool_norm=True)


    # STUFF PLOTTEN veraltet

    # dict_dif = MyAlgorithm.load_data(filename_emb)
    # values_list = []
    # for value in dict_dif.values():
    #     # Remove the wrapping strings using string slicing
    #     value = value[1:-1]
    #     # Convert the modified string back to a list using eval()
    #     value_list = eval(value)
    #     # Add the list to the big list
    #     values_list.extend(value_list)
    # #print(values_list)
    # #count = values_list.count(1.0)
    #
    # count = 0
    # for value in values_list:
    #     if 1.0 <= value <= 1.0:
    #         count += 1
    # frequency = count / len(values_list)
    #
    # print(frequency)
    # #MyAlgorithm.plot_diffs(dict_dif, filename_emb)
    #MyAlgorithm.plot_diff_sum(dict_dif, filename_emb)