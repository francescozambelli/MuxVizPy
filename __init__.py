from MuxVizPy import leading_eigenv_approx 
from MuxVizPy import build
from MuxVizPy import versatility
from MuxVizPy import topology
from MuxVizPy import utils
from MuxVizPy import mesoscale
from MuxVizPy import percolation
from MuxVizPy import plotMux
from MuxVizPy import visualization

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import find, identity, coo_matrix
import pandas as pd
import graph_tool as gt
from graph_tool import centrality, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust
from tqdm import tqdm
from operator import itemgetter
from functools import reduce
import itertools

import warnings
warnings.filterwarnings("ignore")


class VirusMultiplex():
    def __init__(self, indexes, target_folder, virus_metadata, NEIGH_ORDER=1):
        self.target_folder = target_folder
        self.indexes = indexes
        self.mux_ppi = pd.DataFrame()
        self.virus_list = []
        for i in self.indexes:
            if virus_metadata.loc[i,"neigh_order"]==NEIGH_ORDER:
                virus_path = self.target_folder + virus_metadata.loc[i,"virus"]
                self.virus_list.append(virus_metadata.loc[i,"virus_short"])

                virus_edges = pd.read_csv(virus_path+"/edges.csv", header=0, sep=",")
                virus_nodes = pd.read_csv(virus_path+"/nodes.csv", header=0, sep=",")

                if virus_nodes.shape[0]<=1: continue

                human_prot = virus_nodes[np.any([(virus_nodes["type"]==1), (virus_nodes["type"]==2)], axis=0)]["node"]
                human_ppi = virus_edges[np.all([(virus_edges["V1"].isin(human_prot)), (virus_edges["V2"].isin(human_prot))],axis=0)]

                if human_ppi.shape[0]>0:
                    human_ppi["layer"] = virus_metadata.loc[i,"virus_short"]
                    self.mux_ppi = pd.concat([self.mux_ppi, human_ppi])

        self.mux_ppi = self.mux_ppi.reset_index()


        self.Layers = self.mux_ppi["layer"].unique().shape[0]
        self.layer_map = {self.mux_ppi["layer"].unique()[i]: i for i in range(self.Layers) }

        # mapping node names to ids
        self.unique_nodes = pd.concat([self.mux_ppi["V1"], self.mux_ppi["V2"]]).unique()
        self.node_map = {self.unique_nodes[i]: i for i in range(self.unique_nodes.shape[0]) }

        self.Nodes = self.unique_nodes.shape[0]
        self.Edges = self.mux_ppi.shape[0]

        self.mux_ppi["nodeA"] = itemgetter(*self.mux_ppi["V1"].to_numpy())(self.node_map)
        self.mux_ppi["nodeB"] = itemgetter(*self.mux_ppi["V2"].to_numpy())(self.node_map)
        self.mux_ppi["l"] = itemgetter(*self.mux_ppi["layer"].to_numpy())(self.layer_map)

        #generate an list of networks and aggregate/aggregate-binary network
        self.g_list = []

        for l in range(self.Layers):
            tmp = self.mux_ppi[self.mux_ppi["l"]==l].reset_index()[["nodeA", "nodeB"]]
            tmp = tmp.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
            self.g_list.append(gt.Graph(directed=False))
            self.g_list[l].add_edge_list(tmp.values)
            self.g_list[l].add_vertex(self.Nodes-self.g_list[l].num_vertices())

        self.net_description = "Multiplex with " + str(self.Layers) + " layers, " + str(self.Nodes) + " nodes and " + str(self.Edges) + " edges"

        #generate the mulilayer network structure
        tmp_multi = self.mux_ppi.reset_index()[["nodeA", "nodeB"]]
        tmp_multi = tmp_multi.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
        self.g_multi = gt.Graph(directed=False)
        self.g_multi.add_edge_list(tmp_multi.values)
        ep_weight_lay = []

        for l in range(self.Layers):
            ep_weight_lay.append([l]*self.mux_ppi["l"].value_counts()[l])
        ep_weight_lay = np.concatenate(ep_weight_lay)

        self.g_multi.edge_properties["weight"]=self.g_multi.new_edge_property("int", ep_weight_lay)
                
            
            
class VirusMultiplex_from_dirlist():
    def __init__(self, dirlist):
        self.dirlist = dirlist 
        self.mux_ppi = pd.DataFrame()
        for idir in self.dirlist:
            human_ppi = pd.read_csv(idir+"/edges.csv", header=0, sep=",")
            human_ppi.columns = ["source", "target"]
            human_ppi["layer"] = idir.split("/")[-1]
            self.mux_ppi = pd.concat([self.mux_ppi, human_ppi])

        self.mux_ppi = self.mux_ppi.reset_index()


        self.Layers = self.mux_ppi["layer"].unique().shape[0]
        self.layer_map = {self.mux_ppi["layer"].unique()[i]: i for i in range(self.Layers) }

        # mapping node names to ids
        self.unique_nodes = pd.concat([self.mux_ppi["source"], self.mux_ppi["target"]]).unique()
        self.node_map = {self.unique_nodes[i]: i for i in range(self.unique_nodes.shape[0]) }

        self.Nodes = self.unique_nodes.shape[0]
        self.Edges = self.mux_ppi.shape[0]

        self.mux_ppi["nodeA"] = itemgetter(*self.mux_ppi["source"].to_numpy())(self.node_map)
        self.mux_ppi["nodeB"] = itemgetter(*self.mux_ppi["target"].to_numpy())(self.node_map)
        self.mux_ppi["l"] = itemgetter(*self.mux_ppi["layer"].to_numpy())(self.layer_map)

        #generate an list of networks and aggregate/aggregate-binary network
        self.g_list = []

        for l in range(self.Layers):
            tmp = self.mux_ppi[self.mux_ppi["l"]==l].reset_index()[["nodeA", "nodeB"]]
            tmp = tmp.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
            self.g_list.append(gt.Graph(directed=False))
            self.g_list[l].add_edge_list(tmp.values)
            self.g_list[l].add_vertex(self.Nodes-self.g_list[l].num_vertices())

        self.net_description = "Multiplex with " + str(self.Layers) + " layers, " + str(self.Nodes) + " nodes and " + str(self.Edges) + " edges"

        #generate the mulilayer network structure
        tmp_multi = self.mux_ppi.reset_index()[["nodeA", "nodeB"]]
        tmp_multi = tmp_multi.rename({"nodeA":"source", "nodeB":"target"}, axis=1)
        self.g_multi = gt.Graph(directed=False)
        self.g_multi.add_edge_list(tmp_multi.values)
        ep_weight_lay = []

        for l in range(self.Layers):
            ep_weight_lay.append([l]*self.mux_ppi["l"].value_counts()[l])
        ep_weight_lay = np.concatenate(ep_weight_lay)

        self.g_multi.edge_properties["weight"]=self.g_multi.new_edge_property("int", ep_weight_lay)