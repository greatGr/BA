import csv
import math

import networkx as nx
import numpy as np
import torch
import random
import ast

from matplotlib import pyplot as plt

import MyFeedForward
import MyGraph
import NaiveEmbedding
import Node2VecEmbedding


def compute_all_paths(filename, bool_norm):
    result = filename.split("_")
    graph = MyGraph.load_graph(result[0], result[1])

    if "naiv" in filename:
        embedding = NaiveEmbedding.load_node_emb(filename, normalized=bool_norm)
    else:
        embedding = Node2VecEmbedding.load_node_emb(filename, normalized=bool_norm)

    model = MyFeedForward.load_model(filename)

    dict_paths = {}
    dict_diffs = {}
    for i in range(2, nx.diameter(graph)+1):
        dict_paths[i] = []
        dict_diffs[i] = []

    count_sum = 0
    for start in graph.nodes():
        for ziel in graph.nodes():
            if (start != ziel) and (ziel not in graph[start]):
                #computed_path = compute_path_greedy(graph, start, ziel, model, embedding)
                computed_path = compute_path(graph, start, ziel, model, embedding)
                if computed_path != -1:
                    dist_shortest_path = len(nx.bidirectional_shortest_path(graph, start, ziel)) -1
                    #computed path zu dict paths an richtiger stelle hinzufügen
                    dict_paths[dist_shortest_path] += [computed_path]
                    dict_diffs[dist_shortest_path] += [(len(computed_path) - 1) / dist_shortest_path]
                    #dict_diffs[dist_shortest_path] += [(len(computed_path)-1)/dist_shortest_path]
                else:
                    count_sum += 1

    print("Summe nicht berechneter Wege", count_sum)

    save_data(dict_diffs, filename)
    plot_diffs(dict_diffs, filename)


def compute_path_greedy(G, start, ziel, model, embedding):
    reset_visited(G)
    backtracking_list = []

    #print("kürzester Weg", nx.shortest_path(G, start, ziel))

    path = [start]
    current = start
    G.nodes[current]["visited"] = 1
    while ziel not in G[current]:
        moved = 0
        for n in G[current]:
            if (G.nodes[n]["visited"] == 0) and (n not in backtracking_list):
                input = np.concatenate((embedding[int(current)], embedding[int(n)], embedding[int(ziel)]), axis=0)
                prediction = model(torch.tensor(input))
                G.nodes[n]["visited"] = 1
                if (prediction >= 0.5):
                    path += [n]
                    current = n
                    moved = 1

        #Falls kein Nachbarknoten mit Ja klassifiziert wurde
        if moved == 0:
            #current = random_choice(G, current, path)
            current = backtracking(G, current, path, backtracking_list)

    path += [ziel]

    while len(path) !=len(set(path)):
        remove_circle(path)

    return path

def compute_path(G, start, ziel, model, embedding):

    path = [start]
    current = start
    while ziel not in G[current]:
        predictions_dict = {}
        for n in G[current]:
            if n not in path:
                input = np.concatenate((embedding[int(current)], embedding[int(n)], embedding[int(ziel)]), axis=0)
                pred_n = model(torch.tensor(input))
                predictions_dict[n] = pred_n
        if predictions_dict != {}:
            max_value = max(predictions_dict.values())
            max_key = [k for k, v in predictions_dict.items() if v == max_value][0]
            path += [max_key]
            current = max_key
        else:
            return -1
    path += [ziel]

    return path


def backtracking(G, current, path, backtracking_list):

    if current == path[0]:
        # Backtracking auf dem Startknoten nicht möglich
        backtracking_list += [current]
        #Der nächste Knoten wird zufällig ausgewählt
        next = random_choice(G, current, path)
        return next
    elif (path[-1] in backtracking_list):
        #Auf jetzigem Knoten wurde schonmal backtracking gemacht
        next = random_choice(G, current, path)
        return next
    elif (path[-2] in backtracking_list):
        # Auf vorherigem Knoten wurde schonmal backtracking gemacht
        backtracking_list += [current]
        next = random_choice(G, current, path)
        return next

    #Ganz normaler Fall Backtracking
    backtracking_list += [current]
    for n in G[current]:
        if n not in path:
            G.nodes[n]["visited"] = 0
    del path[-1]
    next = path[-1]

    return next

def random_choice(G, current, path):

    #komplett zufällig einen wählen am Ende Kreise entfernen
    next = random.choice(list(G[current].keys()))
    G.nodes[next]["visited"] = 1
    path += [next]

    return next

#Funktioniert gerade nicht
def max_choice(G, current, path, ziel, model, embedding):
    predictions = []
    for n in G[current]:
        input = np.concatenate((embedding[int(current)], embedding[int(n)], embedding[int(ziel)]), axis=0)
        # print("Input", input)
        prediction = model(torch.tensor(input))
        predictions += [prediction]

    while len(predictions) != 0:
        max_value = max(predictions)
        print(max_value)
        max_index = predictions.index(max_value)
        predictions.remove(max_value)
        next = int(list(G[current].keys())[max_index])
        if str(next) not in path:
            path += [str(next)]
            return str(next)
    print("Problem Kreis")
    return -1

def remove_circle(path):
    duplicates = {}

    for i, num in enumerate(path):
        if num not in duplicates:
            duplicates[num] = [i]
        else:
            duplicates[num].append(i)

    max_distance = 0
    max_dup = None
    for num, indices in duplicates.items():
        if len(indices) > 1:
            distance = max(indices) - min(indices)
            if distance > max_distance:
                max_distance = distance
                max_dup = num

    if max_dup is not None:
        start = min(duplicates[max_dup])
        end = max(duplicates[max_dup])
        del path[start+1:end+1]

def reset_visited(graph):
    for node in graph.nodes():
        graph.nodes[node]["visited"] = 0

def plot_diffs(dict_diffs, filename):

    n_rows = int(math.sqrt(len(dict_diffs)))
    n_cols = math.ceil(len(dict_diffs)/n_rows)
    fig, axs = plt.subplots(nrows = n_rows, ncols = n_cols)
    fig.tight_layout(h_pad=5, w_pad=2)
    fig.set_size_inches(9, 7)
    for i, key in enumerate(dict_diffs):
        diffs = dict_diffs[key]
        if isinstance(diffs, str):
            diffs_list = ast.literal_eval(diffs)
        else:
            diffs_list = diffs

        x_ind = (i % (n_cols))
        y_ind = (i//(n_cols))

        axs[y_ind][x_ind].hist(list(diffs_list), bins=20)
        axs[y_ind][x_ind].set_title(f"Wege Länge {int(key)}")
        axs[y_ind][x_ind].set_xlabel("Abweichung als \n (berechn./tats.Länge)")
        axs[y_ind][x_ind].set_ylabel("Anzahl")

        print("Länge {}: maximale Abweichung {}".format(int(key), max(diffs_list)))
        #print("Länge {}: minimale Abweichung {}".format(key, min(diffs)))


    path = "Abbildungen/Histogramme_Abweichung/" + filename + ".png"
    fig.savefig(path)

def save_data(dict, filename):

    filename_cvs = "Csv Files Computed Paths/" + filename + ".csv"

    with open(filename_cvs, "w", newline="") as csvfile:
        # Define the CSV file writer
        writer = csv.writer(csvfile)

        for key, value in dict.items():
            writer.writerow([key, value])

def load_data(filename):
    path = "Csv Files Computed Paths/" + filename + ".csv"
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile)

        my_dict = dict(reader)

    return my_dict
