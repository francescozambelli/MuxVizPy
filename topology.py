import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust
from functools import reduce
from tqdm import tqdm

from .leading_eigenv_approx import leading_eigenv_approx
from .build import *

def get_multi_LCC(obj, obj_type="glist"):
    if obj_type=="glist":
        tensor = get_node_tensor_from_network_list(obj)
    elif obj_type=="tensor":
        tensor = obj
    agg = get_aggregate_network(tensor)
    
    lcc = gt.topology.extract_largest_component(agg).get_vertices()
    
    return lcc

def get_multi_LIC(obj, obj_type="glist"):
    if obj_type=="glist":
        glist = obj
    elif obj_type=="tensor":
        glist=[]
        for t in tensor:
            g=gt.Graph(directed=False)
            g.add_edge_list(np.transpose(t.todense().nonzero()))
            glist.append(g)
        
    lcc_per_lay = [gt.topology.extract_largest_component(g).get_vertices() for g in glist]
    lcc_inters = reduce(np.intersect1d, lcc_per_lay)
    
    return lcc_inters
    

def get_multi_LVC(g_list, printt=True):
  g_l = [gt.Graph(g) for g in g_list]
  layers = len(g_l)
  lvc = []
  flag_nochange = False

  #set node names, because we want to preserve labels while pruning later
  for l in range(layers):
    names = g_l[l].new_vertex_property("int", g_l[l].get_vertices())
    g_l[l].vp["names"]=names


  iteration = 0
  while len(lvc)==0 or not(flag_nochange):
    iteration = iteration + 1
    if printt: print(" LVC Iteration #", iteration, "...", "\n")
    #calculate the intersection of LCCs
    l = 1
    lic = get_multi_LIC(g_l)

    if len(lvc)==0:
      lvc = lic
      flag_nochange = False
    else:
      if len(lic) == len(lvc):
        if (np.all(np.sort(lic) == np.sort(lvc))):
          #stop the algorithm
          flag_nochange = True
          return np.array(g_l[0].vp["names"].get_array()[lvc])
        else:
          flag_nochange = False
        
      else:
        flag_nochange = False

    if not(flag_nochange):
      #clear each layer from nodes not in the intersection
      for l in range(layers):
        d = np.delete(g_l[l].get_vertices(),lic)
        if len(d)!=0:
          g_l[l].remove_vertex(d)
      lvc = lic

  return np.array(g_l[0].vp["names"].get_array()[lvc])


def get_connected_components(supra, layers, nodes):
    g = gt.Graph(directed=False)
    g.add_edge_list(np.transpose(supra.nonzero()))
    components_all = gt.topology.label_components(g)
    
    components = components_all[0]
    components_size = components_all[1]

    #first we have to check if the same entity is assigned to the same component
    #eg, if node A in layer 1 is assigned to component 1 and node A in layer 2 is assigned to component 2
    #then it makes no sense to collapse the information: if they are a unique entity, the nodes
    #should be assigned to the same component, and this happens if they are interconnected or
    #if some of the replicas are isolated components while the others are interconnected

    if layers > 1:
      new_components = np.zeros(nodes)
      for n in range(nodes): 
        comp = components[n]  #the component assigned to n in layer 1
        new_components[n] = comp

        for l in range(1,layers):
          ctmp = components[l*nodes+n]
          if ctmp!=comp:
            #check if it is isolated
            if components_size[ctmp]!=1 and components_size[comp]!=1:
              print("  Impossible to find meaningful connected components\n")
              print(f"  Node {n} in layer 0 is in component {comp} (size {components_size[comp]}) while ")
              print(f"  Node {n} (abs id: {l*nodes+n}) in layer {l} is in component {ctmp} (size {components_size[ctmp]}) \n")
              raise ValueError("  Aborting process.\n")

      components = np.zeros(nodes)
      comps = np.unique(new_components)

      #readjust the components label
      for i in range(len(comps)):
        components[np.where(new_components==comps[i])]=i

    return components
    
def get_multi_path_statistics(supra, layers, nodes):
    n = supra.shape[0]

    if layers==1:
        #standard monoplex analysis
        #g <- igraph::graph_from_adjacency_matrix(SupraAdjacencyMatrix, weighted=T, mode="undirected")
        g = gt.Graph(directed=False)
        g.add_edge_list(np.transpose(supra.nonzero()))
             
        DMIN = gt.topology.shortest_distance(g).get_2d_array(g.get_vertices())
        
    else:
        #multilayer analysis
        #g.ext <- igraph::graph_from_adjacency_matrix(SupraAdjacencyMatrix, weighted=T, mode="undirected")
        #TODO: I removed the restriction to undirected mode only, worth checking that it does not create issues
        g_ext = gt.Graph(directed=False)
        g_ext.add_edge_list(np.transpose(supra.nonzero()))
  
        DMIN=np.zeros([nodes,nodes])
        
        for j in tqdm(range(nodes)):
        
            for k in range(layers):
                DP=gt.topology.shortest_distance(g_ext, source=g_ext.vertex((k*nodes)+j)).get_array()
                DP_min=np.minimum.reduce(DP.reshape([layers,nodes]))
                if k==0:
                    DMIN[j,:]=DP[:nodes]
                DMIN[j,:]=np.minimum(DMIN[j,:], DP_min)

    ###############
    #closeness
    ###############
    #Opsahl, T., Agneessens, F., Skvoretz, J. (2010). Node centrality in weighted networks: Generalizing degree and shortest paths. Social Networks 32, 245-251
    #https://toreopsahl.com/2010/03/20/closeness-centrality-in-networks-with-disconnected-components/

    closeness = [np.mean(1/np.delete(DMIN[n,:],n)) for n in range(nodes)]
    #0 if disconnected, 1 if connected to all other nodes

    ###############
    #avg path length
    ###############
    #this is a personal definition, should be checked.
    #of course, for networks with isolated nodes mean(1/closeness) = Inf, instead the following is safe:
    avg_path_length = 1 / np.mean(closeness)

    ###############
    #betweenness
    ###############
    #todo

    return {"distance_matrix":DMIN,
            "avg_path_length":avg_path_length,
            "closeness":closeness}
            
            
            
def get_SP_similarity_matrix(supra, layers, nodes):
    
    distance_list = []

    g_list = supra_adjacency_to_network_list(supra, layers, nodes)

    for l in range(layers):
      distance_list.append(gt.topology.shortest_distance(g_list[l]).get_2d_array(g_list[l].get_vertices()))
      distance_list[l][distance_list[l]>1e8] = 1e8

    frobenius_norm = np.zeros([layers,layers])
    for l1 in range(layers-1):
      for l2 in range(l1,layers):
        frobenius_norm[l1,l2] = np.linalg.norm(distance_list[l1]-distance_list[l2])
        frobenius_norm[l2,l1] = frobenius_norm[l1,l2]

    frobenius_norm = 1-(frobenius_norm/np.max(frobenius_norm))

    return frobenius_norm
