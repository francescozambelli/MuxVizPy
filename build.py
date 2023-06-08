import numpy as np
import scipy as sp
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust

from .leading_eigenv_approx import leading_eigenv_approx

def speye(n):
    return sps.eye(n, format="csr")

def build_layers_tensor(Layers, OmegaParameter, MultisliceType):
    MultisliceType = MultisliceType.lower()
    M = sps.csr_matrix((Layers, Layers))
    if Layers > 1:
        if MultisliceType == "ordered":
            M = (sps.diags([np.ones(Layers-1), np.ones(Layers-1)], [1, -1]))*OmegaParameter
        elif MultisliceType == "categorical":
            M = (np.ones((Layers, Layers)) - np.eye(Layers))*OmegaParameter
        elif MultisliceType == "temporal":
            M = sps.diags([np.ones(Layers-1)], [1])*OmegaParameter
    else:
        M = 0
        print("--> Algorithms for one layer will be used")
    return M


def build_supra_adjacency_matrix_from_edge_colored_matrices(nodes_tensor, layer_tensor, layers, nodes):
    identity_mat = speye(nodes)
    M = sps.block_diag(nodes_tensor)+sps.kron(layer_tensor, identity_mat)
    return M

def get_node_tensor_from_supra_adjacency(supra, layers, nodes):
    tensor = []
    for i in range(layers):
        tensor.append(supra[i*nodes:(i+1)*nodes,i*nodes:(i+1)*nodes])
    return tensor

def get_node_tensor_from_network_list(glist):
    tensor = []
    for g in glist:
        tensor.append(gt.spectral.adjacency(g))
    return tensor

def get_aggregate_network(obj, obj_type="tensor", return_mat=False, binarize=True):
    
    if obj_type=="glist":
        tensor = get_node_tensor_from_network_list(obj)
        obj = tensor
    
    if obj_type=="tensor" or obj_type=="glist":
        agg_mat = sps.coo_matrix(obj[0].shape)
        for i in range(len(obj)):
            agg_mat+=obj[i]
        
        if binarize:
            agg_mat[agg_mat>0]=1
        
        if return_mat:
            return agg_mat
        else:
            g_agg = gt.Graph(directed=False)
            g_agg.add_edge_list(np.transpose(agg_mat.nonzero()))
            g_agg.add_vertex(agg_mat.shape[0]-g_agg.num_vertices())
            return g_agg
            

def supra_adjacency_to_block_tensor(supra, layers, nodes):
    #this is equivalent to Matlab's   BlockTensor = {}
    BlockTensor = []

    for i in range(layers):
        BlockTensor.append([])
        for j in range(layers):
            BlockTensor[i].append(supra[i*nodes:(i+1)*nodes, j*nodes:(j+1)*nodes])
    return BlockTensor
   

def node_tensor_to_network_list(tensor, layers, nodes):
    g_list=[]
    for i in range(layers):
        g=gt.Graph(directed=False)
        g.add_edge_list(np.transpose(tensor[i].nonzero()))
        g_list.append(g)
    return g_list
   
    
def supra_adjacency_to_network_list(supra, layers, nodes):
    node_tensor = get_node_tensor_from_supra_adjacency(supra, layers, nodes)
    g_list = node_tensor_to_network_list(node_tensor, layers, nodes)
    return g_list
    

def get_laplacian_matrix(g):
    adj_mat = gt.spectral.adjacency(g)
    N = adj_mat.shape[0]
    u = sps.coo_matrix(np.ones([N,1]))
    
    adj_dot = adj_mat.dot(u)
    
    lap_mat = sps.diags(adj_dot.toarray().flatten())-adj_mat
    
    if ((lap_mat.dot(u)).sum())>1e-8:
        raise ValueError("ERROR! The Laplacian matrix has rows that don't sum to 0. Aborting process.")
    
    lap_mat.eliminate_zeros()
    return lap_mat

def build_supra_transition_matrix_from_supra_adjacency_matrix(supra, layers, nodes, Type = "classical"):
    order = layers*nodes
    supra_strength = supra.sum(axis=1).flatten()
    disconnected_nodes = order - len(supra_strength.nonzero()[1])
    in_layer_disconnected_count = 0

    if disconnected_nodes > 0:
        print(" #Trapping nodes (no outgoing-links): ",disconnected_nodes,"\n")

    if Type == "pagerank" or Type == "classical":
        supra_strength[0,np.array(supra_strength>0)[0]] = 1. / supra_strength[0,np.array(supra_strength>0)[0]]
        supra_strength = sps.diags(np.array(supra_strength)[0])
        supra_transition = supra_strength.dot(supra)

    if Type == "pagerank":
        alpha = 0.85
    elif Type == "classical":
        alpha = 1
    else:
        # all other types, i.e. diffusive | maxent | physical
        alpha = 1
    
    if (disconnected_nodes > 0):
        ids = np.where(supra_transition.sum(axis=1) == 0)[0]

        if Type == "pagerank":
            print(" #Using uniform teleportation from nodes with no outgoing links\n")
            """
            #to normalize correctly in the case of nodes with no outgoing links:
            supra_transition[0,ids] = 1./order
            not_ids = np.delete(np.arange(len(supra_transition)), ids)
            supra_transition[0,not_ids] = alpha * supra_transition[0,not_ids] + (1-alpha)/order
            """
            #supra_transition.setdiag(1)
        if Type == "classical":
            print(" #Using self-loops for isolated nodes\n")
            #supra_transition[:,ids] = 1./order
            supra_transition[ids,ids]=[1]*len(ids)
    else:
        supra_transition = supra_transition# + sps.coo_matrix(np.zeros(supra_transition.shape))*(1-alpha)/order

    # check stochasticity of transition matrix, equivalently
    # if sum_k (sum_l supraA[k, l]) == NL
    if (abs(supra_transition.sum() - order) > 1e-6):
        raise ValueError("BuildSupraTransitionMatrixFromSupraAdjacencyMatrix: ERROR! Problems in building the supra-transition matrix. Aborting process.")
    
    supra_transition.eliminate_zeros()
    return supra_transition.T


def build_supra_adjacency_matrix_from_extended_edgelist(dfEdges, Layers, Nodes, isDirected):
    """"
    The input must be a pandas dataframe containing the edges between nodes also in different layers.
    The columns name should be NodeIN LayerIN NodeOUR LayerOUT
    """
    
    if len(pd.unique(pd.concat([dfEdges["LayerIN"], dfEdges["LayerOUT"]]))) != Layers:
        raise ValueError("Error: expected number of layers does not match the data. Aborting process.")

    edges = pd.DataFrame({"from": dfEdges["NodeIN"]*(dfEdges["LayerIN"]+1),
                          "to": dfEdges["NodeOUT"]*(dfEdges["LayerOUT"]+1)})
    edges["weight"]=[1]*len(edges)
    
    M = sps.coo_matrix((list(edges["weight"]), zip(*list(edges[["from","to"]].to_numpy()))), shape=(Nodes*Layers, Nodes*Layers))
    
    
    if isDirected==False:
        M = M+M.T

    if abs((M - M.T).sum()) > 1e-12 and isDirected == False:
        raise ValueError("WARNING: The input data is directed but isDirected=FALSE, I am symmetrizing by average.")
        M = (M + M.T) / 2
    
    return M.T


def create_supra_transition_matrix_virus(supra, node_tensor, nodes, layers, p_intra = 1):
    
    supra_sum = np.array(supra.sum(axis=0).tolist()[0])
    supra_nonz = supra.nonzero()
    
    blocks = []
    norms=[]

    #create the non_diagonal blocks: the diagonal entries are the number of non non-itrelayer connections of the nodes he can reach
    for l in range(layers):
        # i subtract (layers-1) to delete the contributions of the interlayer connections given by the categorical coupling
        block = sps.identity(nodes).multiply(supra_sum[l*nodes:(l+1)*nodes]-(layers-1))
        blocks.append(block)
    
    #the blocks are combined together and the rows are normalized to create a transition matrix without diagonal blocks
    mat =[]
    for la in range(layers):
        norm_fac = np.sum(supra_sum.reshape(layers, nodes)[np.delete(np.arange(layers),la)]-(layers-1), axis=0)
        norm_fac = np.where(norm_fac != 0, 1 / norm_fac, 0)
        mat.append([blocks[i].multiply(norm_fac) for i in np.delete(np.arange(layers), la)])
    
    #creating diagonal blocks by applying the standard prodecure for trans matr to each adj matrix
    diag_blocks = []
    for la in range(layers):
        t0_sum = np.array(list(node_tensor[la].sum(axis=0)))[0][0]
        t0_sum = np.where(t0_sum != 0, 1 / t0_sum, 0)

        diag_blocks.append(node_tensor[la].dot(sps.diags(t0_sum)).T)
    
    #combine the results in the unnormalized final matrix
    #the sum of each row can be 2, if there are contribution both from intra e inter connections, 1 for only intra, and 0 for nodes that becomes disconnected
    #if I want to give an exctra contribution to the intra connections, i should choose p_intra>1
    for i in range(layers):
        mat[i].insert(i, diag_blocks[i].multiply(np.array([p_intra]*nodes)))

    comb_mat =sps.vstack([sps.hstack(mat[i]) for i in range(layers)])
    
    valss = np.where(comb_mat.sum(axis=1) != 0, 1 / comb_mat.sum(axis=1), 0).flatten()

    return comb_mat.T.multiply(valss).T