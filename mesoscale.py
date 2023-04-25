import numpy as np
import scipy as sp
from tqdm import tqdm
from graph_tool import inference

def get_mod(g_multi, n_iter, return_state=False):
    modules_list = []
    modularity_list = []

    for It_Com in tqdm(range(n_iter)):
        state_multi = inference.minimize_blockmodel_dl(g_multi,
                                                          state_args=dict(base_type=inference.LayeredBlockState,
                                                          state_args=dict(ec=g_multi.ep.weight, layers=True)))

        modules_list.append(state_multi.get_nonempty_B())
        modularity_list.append(inference.modularity(g_multi, state_multi.get_blocks()))
    if return_state:
        return [modules_list, modularity_list, state] 
    else:
        return [modules_list, modularity_list]


def inter_layer_assortativity(g_list, layers):
    degrees = [g_list[i].get_total_degrees(np.arange(g_list[i].num_vertices())) for i in range(layers)]
    pearson_ass = np.zeros([layers, layers])
    spearman_ass = np.zeros([layers, layers])
    
    for l1 in range(layers):
        for l2 in range(layers):
            pearson_ass[l1,l2] = sp.stats.pearsonr(degrees[l1], degrees[l2]).statistic
            spearman_ass[l1,l2] = sp.stats.spearmanr(degrees[l1], degrees[l2]).statistic
    return {"Pearson": pearson_ass, "Spearman": spearman_ass}
    