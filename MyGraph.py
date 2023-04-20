import numpy as np
from matplotlib import pyplot as plt
from networkx import diameter
from numpy import random
from scipy.spatial import Delaunay
import networkx as nx

#Methode erstellt Delaunay Graph mit x Knoten
def new_graph(number_nodes, filename):

    #2d-numpy Array mit den Koordinaten der zufälligen Punkte im Einheitsquadrat
    coordinates_nodes = new_rand_points(number_nodes)

    #Berechnet die Delaunay-Triangulation der Punkte in coordinates_nodes
    del_triangulation = delaunay_triangulation(coordinates_nodes)

    #Graph mit den Knoten und der Delaunay-Triangulation erstellen
    graph = create_graph(del_triangulation, coordinates_nodes)

    # Graph speichern
    save_graph(graph, filename)
    graph = load_graph(filename)

    #AUSGABE: Der Delaunay Graph
    return graph

#Methode speichert Graph im GML Format in Datei
def save_graph(graph, file_name):

    #Pfad wo die Datei gespeichert wird
    path = "Graphen/" + file_name
    #Speichert Adjazenzliste zu Graph in die Datei
    nx.write_gml(graph, path, stringizer=str)

#Methode lädt im GML Format gespeicherten Graph aus Datei
def load_graph(file_name):

    #Pfad dahin wo die Datei gespeichert ist
    path = "Graphen/" + file_name
    #Lädt Daten aus Datei und erzeugt entsprechenden Graph
    graph = nx.read_gml(path, destringizer=str)

    #AUSGABE: Der aus der Datei geladene Graph
    return graph

#Erstellt ein 2d-numpy Array mit number Einträgen
def new_rand_points(number):

    #Erstellt numpy Array der Größe (number,2) mit zufälligen Punkten im Einheitsquadrat
    coordinates = random.rand(number, 2)

    #print(coordinates.shape)

    #AUSGABE: 2d-numpy Array mit den Koordinaten der zufälligen Punkte
    return coordinates

#Berechnet die Delaunay-Triangulation der Knoten
def delaunay_triangulation(coordinates_nodes):

    #Delaunay-Triangulation zu coordinates_nodes berechnen
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html
    tri = Delaunay(coordinates_nodes)

    #print("unused points in triangulation:", tri.coplanar)

    #AUSGABE:
    return tri

#Erstellt einen Delaunay Graph
def create_graph(tri, coordinates):

    #Menge für alle Kanten des Graphen erstellen
    edges = set()

    #Alle Dreiecke der Triangulation druchgehen und Kanten zu Menge hinzufügen
    for x in range(tri.nsimplex):
        edge = sorted([tri.simplices[x, 0], tri.simplices[x, 1]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.simplices[x, 0], tri.simplices[x, 2]])
        edges.add((edge[0], edge[1]))
        edge = sorted([tri.simplices[x, 1], tri.simplices[x, 2]])
        edges.add((edge[0], edge[1]))

    #Graph aus der Liste der Kanten erstellen
    graph = nx.Graph(list(edges))

    #Dictionary erstellen mit den Knoten als Keys und ihren Koordinaten als Einträgen
    x_position = dict(zip(range(len(coordinates)), coordinates[:,0]))
    y_position = dict(zip(range(len(coordinates)), coordinates[:,1]))

    #Dictionary erstellen mit den Knoten als Keys und dem "visited status" als Eintrag, alle nicht besucht
    visited = dict(zip(range(len(coordinates)), [0] * graph.number_of_nodes()))
    #Dictionary erstellen mit den Kanten als Keys und ihrem count status als Eintrag, überall 0
    counted = dict(zip(list(edges), [0] * graph.number_of_edges()))

    #Koordinaten als Attribute zu den Knoten hinzufügen
    nx.set_node_attributes(graph, x_position, "x_pos")
    nx.set_node_attributes(graph, y_position, "y_pos")
    #Visited Status als Attribut hinzufügen
    nx.set_node_attributes(graph, visited, "visited")
    #Edge count als Kanten Attribut hinzufügen
    nx.set_edge_attributes(graph, counted, "count")

    #AUSGABE: Der Delaunay-Graph
    return graph

def plot_graph(graph, filename):
    # Dictionary erstellen mit den Knoten als Keys und ihren Positionen als Einträgen
    x_pos = nx.get_node_attributes(graph, "x_pos")
    y_pos = nx.get_node_attributes(graph, "y_pos")
    pos = dict()
    for i in x_pos:
        pos[i] = [x_pos[i], y_pos[i]]

    # Knoten des Graphen malen
    plt.subplot(121, title="Delaunay Graph")
    nx.draw_networkx_nodes(graph, pos, node_size=50, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos)
    # Kanten malen
    nx.draw_networkx_edges(graph, pos)

    # Abbildung speichern
    path = "Abbildungen/Del Graphen" + filename
    plt.savefig(path)

