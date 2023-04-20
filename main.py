import networkx as nx

import MyDataset
import MyFeedForward
import MyGraph
import Node2VecEmbedding
from MyDataset import make_datasets
from MyGraph import new_graph
from NaiveEmbedding import compute_embedding

if __name__ == "__main__":

    #Parameter Graph
    nodes = 128

    filename_graph = str(nodes)

    # Erstellen Graph
    G = MyGraph.new_graph(nodes, filename_graph)
    #G = MyGraph.load_graph(str(nodes)
    print("Diameter: ", nx.diameter(G))

    # Naive Einbettung der Knoten
    filename_emb_naiv = filename_graph + "_naiv"
    embedding_naiv = compute_embedding(G, filename_emb_naiv, norm=True)

    # Parameter Node2Vec Einbettung
    dim = nodes
    l_walks = int(nx.diameter(G) * 1.5)
    n_walks = 10
    param_p = 0.1
    param_q = 10
    window_size = 1

    filename_emb_n2v = str(nodes) +"_"+ str(dim) +"_"+ str(param_p) +"_"+ str(param_q) +"_"+ str(window_size)
    Node2VecEmbedding.make_embedding(G, filename_emb_n2v, dim, l_walks, n_walks, param_p, param_q, window_size)

    # Trainings-und Testdaten erstellen
    filename_data_naiv = filename_emb_naiv
    filename_data_n2v = filename_emb_n2v
    filename_paths = filename_graph + "_asp"

    # Listen (Anzahl Startknoten, Länge der Wege)
    l_train = []
    for i in range(2, nx.diameter(G)+1):
        l_train += [(nodes, i)]
    l_test = []

    MyDataset.make_datasets(G, l_train, l_test, filename_emb_naiv, filename_data_naiv, emb_type="naiv", normalized=True)
    MyDataset.make_datasets(G, l_train, l_test, filename_emb_n2v, filename_data_n2v, emb_type="n2v", normalized=True)

    #Parameter für Training neuronales Netz
    data_split = 0.7
    dim_emb = nodes
    list_hidden = [64, 16, 4]
    learning_rate = 0.05
    num_epochs = 10

    #Neuronales Netz mit naiver Einbettung trainieren
    MyFeedForward.train_classifier(filename_data_naiv, data_split, dim_emb, list_hidden, learning_rate, num_epochs)
    #Neuronales Netz mit n2v Einbettung trainieren
    MyFeedForward.train_classifier(filename_data_n2v, data_split, dim_emb, list_hidden, learning_rate, num_epochs)

