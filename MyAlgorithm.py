import csv
import math

import networkx as nx
import numpy as np
import torch
import random
import ast

from matplotlib import pyplot as plt
from scipy.spatial import distance

import ForceEmbedding
import MyDataset
import MyFeedForward
import MyGraph
import NaiveEmbedding
import Node2VecEmbedding


def compute_all_paths(filename, bool_norm):
    result = filename.split("_")
    graph = MyGraph.load_graph(result[0], result[1])

    if "naiv" in filename:
        embedding = NaiveEmbedding.load_node_emb(filename, normalized=bool_norm)
        model = MyFeedForward.load_model(filename)
    elif "force" in filename:
        embedding = ForceEmbedding.load_node_emb(filename, normalized=bool_norm)
    else:
        embedding = Node2VecEmbedding.load_node_emb(filename, normalized=bool_norm)
        model = MyFeedForward.load_model(filename)

    #Dictionary (2:[[],[]...) key: länge der Wege, value: liste mit allen wegen dieser länge
    #vor kreisentfernung
    dict_paths = {}
    #nach kreisentfernung
    dict_paths_no_circ = {}
    #Dictionary key: länge der Wege, value: liste mit der differenz zwischen berechneter und tatsächlichr länge aller wege
    #vor kreisentfernung
    dict_diffs = {}
    #nach kreisentfernung
    dict_diffs_no_circ = {}

    for i in range(2, nx.diameter(graph)+1):
        dict_paths[i] = []
        dict_paths_no_circ[i] = []
        dict_diffs[i] = []
        dict_diffs_no_circ[i] = []

    count_sum = 0
    for start in graph.nodes():
        for ziel in graph.nodes():
            if (start != ziel) and (ziel not in graph[start]):
                if "force" in filename:
                    computed_path, path_no_circ = compute_path_tri(graph, start, ziel, embedding, filename)
                    #list nicht berechneter Wege
                    if computed_path == -1:
                        count_sum += 1
                        continue
                else:
                    computed_path, path_no_circ = compute_path_nn(graph, start, ziel, model, embedding, filename)
                    #Liste nicht berechneter Wege
                    if computed_path == -1:
                        count_sum += 1
                        continue

                dist_shortest_path = len(nx.bidirectional_shortest_path(graph, start, ziel)) -1

                #computed path zu dict paths an richtiger stelle hinzufügen
                dict_paths[dist_shortest_path] += [computed_path]
                dict_paths_no_circ[dist_shortest_path] += [path_no_circ]
                dict_diffs[dist_shortest_path] += [(len(computed_path) - 1) / dist_shortest_path]
                dict_diffs_no_circ[dist_shortest_path] += [(len(path_no_circ) - 1) / dist_shortest_path]

    save_data(dict_diffs, "diffs/" + filename)
    save_data(dict_diffs_no_circ, "diffs_no_circ/" + filename)
    save_data(dict_paths, "paths/" + filename)
    save_data(dict_paths_no_circ, "paths_no_circ/" + filename)

    print("nicht berechnet", count_sum)

    #plot_diffs(dict_diffs, filename)

#Berechnet Weg mit Dreiecksungleichung für force embedding
def compute_path_tri(G, start, ziel, embedding, filename):
    print(start, ziel)
    path_edges = []
    node_colors = {}
    for node in G.nodes():
        if int(node) == start or int(node) == ziel:
            node_colors[node] = "blue"
        else:
            node_colors[node] = "white"

    path = [start]
    current = start

    while ziel not in G[current]:

        predictions_dict = {}
        # Nachbarknoten des aktuellen Knotens klassifizieren und wahrscheinlichkeit speichern
        for n in G[current]:
            dist_s_n = euclidean_distance(embedding[current], embedding[n])
            dist_n_t = euclidean_distance(embedding[n], embedding[ziel])
            pred_n = dist_s_n + dist_n_t
            predictions_dict[n] = pred_n

        #nicht zum vorherigen knoten zurückgehen
        # if len(path) >= 2:
        #     if path[-2] in predictions_dict:
        #         del predictions_dict[path[-2]]
        #dafür sorgen dass er sich nach einem kreis anders entscheidet wenn er an gleicher stelle ist
        list_i = find_duplicate_indices(path, current)
        if len(list_i) > 1:
            for i in list_i[:-1]:
                if path[i+1] in predictions_dict:
                    del predictions_dict[path[i+1]]

        if predictions_dict != {}:
            min_value = min(predictions_dict.values())
            min_key = [k for k, v in predictions_dict.items() if v == min_value][0]
            path += [min_key]

            path_edges += [{current, min_key}]
            vorher = nx.shortest_path_length(G, source=current, target=ziel)
            nachher = nx.shortest_path_length(G, source=min_key, target=ziel)

            current = min_key

            #prüfen ob entschidung gut, mittel oder schlecht war
            #gut grün, knoten liegt auf kürzestem weg
            #mitteln orange, weg ist genauso lang wie vorher
            #schlecht rot weg ist länger geworden
            if (vorher - nachher) == 1:
                #richtige entscheidung also grün
                node_colors[current] = "green"
            elif (vorher - nachher) == 0:
                #mittel orange
                node_colors[current] = "orange"
            elif (vorher - nachher) < 0:
                #schlecht rot
                print(start, ziel, "roooooot")
                node_colors[current] = "red"
            else:
                print("Fehler bei Wege Entscheidungen einfärben")

        else:
            return -1, -1

    path_edges += [{current, ziel}]
    path += [ziel]

    #nur speichern wenn der berechnete weg um mehr als 1 länger als der tatsächlich kürzeste Weg ist
    if (len(path)-1) > nx.shortest_path_length(G, source=start, target=ziel) + 1:
        #kürzesten Weg einzeichnen
        edge_colors = []
        for edge in G.edges():
            if set(edge) in path_edges:
                edge_colors += ["red"]
            else:
                edge_colors += ["black"]
        pos = {node: (G.nodes[node]['x_pos'], G.nodes[node]['y_pos']) for node in G.nodes()}
        plt.clf()
        nx.draw_networkx(G, pos, node_color=list(node_colors.values()), edge_color=edge_colors, with_labels=True, node_size=100)
        path_file = "Abbildungen/Wege Entscheidungen/" + str(start) + "_" + str(ziel) + ".png"
        plt.savefig(path_file)

    #kreise entfernen
    path_no_circ = path
    while len(path) != len(set(path)):
        path_no_circ = remove_circle(path)

    #print("test no circ", path, path_no_circ)
    return path, path_no_circ

def euclidean_distance(vec_a, vec_b):
    tmp = 0
    for i in range(len(vec_a)):
        tmp += (vec_b[i]-vec_a[i])**2
    dist_a_b = math.sqrt(tmp)

    return dist_a_b

#Berechenet Weg mit nn als ranker für node2vec und naiv embedding
def compute_path_nn(G, start, ziel, model, embedding, filename):

    path = [start]
    current = start

    while ziel not in G[current]:
        predictions_dict = {}
        #Noch nicht klassifizierte Nachbarknoten des aktuellen Knotens klassifizieren und wahrscheinlichkeit speichern
        for n in G[current]:
            if n not in predictions_dict:
                input = np.concatenate((embedding[int(current)], embedding[int(n)], embedding[int(ziel)]), axis=0)
                pred_n = model(torch.tensor(input))
                predictions_dict[n] = pred_n

        # dafür sorgen dass er sich nach einem kreis anders entscheidet wenn er an gleicher stelle ist
        list_i = find_duplicate_indices(path, current)
        if len(list_i) > 1:
            for i in list_i[:-1]:
                del predictions_dict[path[i + 1]]

        if predictions_dict != {}:
            max_value = max(predictions_dict.values())
            max_key = [k for k, v in predictions_dict.items() if v == max_value][0]
            path += [max_key]
            current = max_key
        else:
            return -1, -1

    path += [ziel]

    # kreise entfernen
    path_no_circ = path
    while len(path_no_circ) != len(set(path_no_circ)):
        path_no_circ = remove_circle(path_no_circ)

    return path, path_no_circ

#Entfernt aus den fertig berechneten Wege alle Kreise
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

    return path

def reset_visited(graph):
    for node in graph.nodes():
        graph.nodes[node]["visited"] = 0

def find_duplicate_indices(lst, number):
    indices = []
    for index, item in enumerate(lst):
        if item == number:
            indices.append(index)
    return indices

#Todo!!!
#def draw_path_colored(G, path)

def plot_diffs(dict_diffs, filename):
    print(dict_diffs)
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
        axs[y_ind][x_ind].set_xlabel("Abweichung")
        axs[y_ind][x_ind].set_ylabel("Häufigkeit")

        print("Länge {}: maximale Abweichung {}".format(int(key), max(diffs_list)))
        #print("Länge {}: minimale Abweichung {}".format(key, min(diffs)))


    path = "Abbildungen/Histogramme_Abweichung/" + filename + ".png"
    fig.savefig(path)

def plot_diff_sum(dict_diffs, filename):
    values_list = []
    for value in dict_diffs.values():
        # Remove the wrapping strings using string slicing
        value = value[1:-1]
        # Convert the modified string back to a list using eval()
        value_list = eval(value)
        # Add the list to the big list
        values_list.extend(value_list)

    print("Anzahl wege ", len(values_list))
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    #
    # ax1.plot(x_epoch, y_acc['train'], 'bo', label='train', ms=2)
    #
    # ax1.set_xlabel("Abweichung")
    # ax1.set_ylabel("Anzahl")

    # Count the frequency of each number in the list
    frequency = {}
    for num in values_list:
        frequency[num] = frequency.get(num, 0) + 1

    # Sort the numbers in ascending order
    numbers = sorted(frequency.keys())
    print("Konplett richtig", frequency[1.0])
    #Alle Abweichungen plotten
    counts = [frequency[num] for num in numbers]
    #Alle außer Abweichung 1
    #counts = [frequency[num] for num in numbers[1:]]

    # Plot the frequency distribution
    plt.scatter(numbers, counts, marker='o', s=4)
    plt.xlabel('Abweichung')
    plt.ylabel('Häufigkeit')
    path = "Abbildungen/Histogramme_Abweichung/" + filename + "neu.png"
    plt.savefig(path)

def draw_frequency_curve(data):
    # Count the frequency of each number in the list
    frequency = {}
    for num in data:
        frequency[num] = frequency.get(num, 0) + 1

    # Sort the numbers in ascending order
    numbers = sorted(frequency.keys())
    counts = [frequency[num] for num in numbers]

    # Plot the frequency distribution
    plt.plot(numbers, counts, marker='o')
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Frequency Distribution')
    plt.show()

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



#NICHT VERWENDET
#NN klassifiziert Knoten, erster Knoten mit ja genommen
def compute_path_greedy_nn(G, start, ziel, model, embedding):
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
def random_choice(G, current, path):

    #komplett zufällig einen wählen am Ende Kreise entfernen
    next = random.choice(list(G[current].keys()))
    G.nodes[next]["visited"] = 1
    path += [next]

    return next
def go_back(G, path, current, ziel, embedding, model):
    print("current", current)
    del path[-1]
    old = current
    current = path[-1]
    predictions_dict = {}
    for n in G[current]:
        if n not in path:
            input = np.concatenate((embedding[int(old)], embedding[int(n)], embedding[int(ziel)]), axis=0)
            pred_n = model(torch.tensor(input))
            predictions_dict[n] = pred_n
    filtered_dict = {}
    for key, value in predictions_dict.items():
        if value < predictions_dict[old]:
            filtered_dict[key] = value
    if filtered_dict != {}:
        max_value = max(filtered_dict.values())
        max_key = [k for k, v in filtered_dict.items() if v == max_value][0]
        current = max_key
        path += [current]
    else:
        return -1

    return current