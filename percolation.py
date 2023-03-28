import numpy as np
import graph_tool as gt
from .build import *
from .versatility import *

def get_percolation(g_list, layers, nodes, method = "pagerank", intralink_strenght=1):
    tensor = get_node_tensor_from_network_list(g_list)
    g_agg = get_aggregate_network(tensor)
    
    if not(method=="pagerank" or method=="betweenness" or method=="multi_classical" or method=="multi_pagerank" or method=="multi_eigenvector"):
        print("Method not implemented, using pagerank on aggregate matrix ordering")
        method="pagerank"
    
    if method == "pagerank":
        order = np.argsort(gt.centrality.pagerank(g_agg).get_array())
    elif method == "betweenness":
        order = np.argsort(gt.centrality.betweenness(g_agg)[0].get_array())
        
    if method =="multi_pagerank" or method == "multi_classical" or method=="multi_eigenvector":
        layerTensor =build_layers_tensor(Layers=layers,
                                         OmegaParameter=intralink_strenght, 
                                         MultisliceType="categorical")

        supra=build_supra_adjacency_matrix_from_edge_colored_matrices(nodes_tensor=tensor,
                                                                      layer_tensor=layerTensor,
                                                                      layers=layers,
                                                                      nodes=nodes)
        
        sup_tra = build_supra_transition_matrix_from_supra_adjacency_matrix(supra, layers, nodes, Type="classical")
        
        if method=="multi_pagerank":
            order = np.argsort(get_multi_RW_centrality(sup_tra, layers, nodes, Type="pagerank"))
        elif method=="multi_classical":
            order = np.argsort(get_multi_RW_centrality(sup_tra, layers, nodes, Type="classical"))
        elif method=="multi_eigenvector":
            order = np.argsort(get_multi_eigenvector_centrality(supra, layers, nodes))
                               
    
    perc_agg_1 = gt.topology.vertex_percolation(g_agg, order)[0]
    perc_agg_2 = gt.topology.vertex_percolation(g_agg, order, second=True)[0]
    max_perc = np.argmax(perc_agg_2)/len(perc_agg_1)

    return {"1ComponentSize": perc_agg_1, "2ComponentSize": perc_agg_2, "CritPoint": max_perc}
