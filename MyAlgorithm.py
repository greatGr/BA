import MyFeedForward


def compute_all_paths(graph, filename_model):

    model = MyFeedForward.load_model(filename_model)

    #for start in graph.nodes():
        #for ziel in graph.nodes():
            #if start != ziel:
                