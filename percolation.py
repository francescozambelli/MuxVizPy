import numpy as np
import graph_tool as gt
from .build import *
from .versatility import *

def get_percolation(g_list, layers, nodes, order):
    tensor = get_node_tensor_from_network_list(g_list)
    g_agg = get_aggregate_network(tensor)
    
    perc_agg_1 = gt.topology.vertex_percolation(g_agg, order)[0]
    perc_agg_2 = gt.topology.vertex_percolation(g_agg, order, second=True)[0]
    max_perc = np.argmax(perc_agg_2)/len(perc_agg_1)

    return {"1ComponentSize": perc_agg_1, "2ComponentSize": perc_agg_2, "CritPoint": max_perc}
