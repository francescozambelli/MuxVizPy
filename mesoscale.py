import numpy as np
from tqdm import tqdm
from graph_tool import inference

def get_mod(g_multi, n_iter):
    modules_list = []
    modularity_list = []

    for It_Com in tqdm(range(n_iter)):
        state_multi = inference.minimize_blockmodel_dl(g_multi,
                                                          state_args=dict(base_type=inference.LayeredBlockState,
                                                          state_args=dict(ec=g_multi.ep.weight, layers=True)))

        modules_list.append(state_multi.get_nonempty_B())
        modularity_list.append(inference.modularity(g_multi, state_multi.get_blocks()))

    return [modules_list, modularity_list] 