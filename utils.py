import numpy as np

def writeComponent(fname, ensemble):
    with open(fname, "w") as cio:
        for i in range(len(ensemble)):
            cio.write(" ".join(map(str,ensemble[i]))+"\n")

def readComponent(fname):
    with open(fname, "r") as fread:
        read_list = fread.readlines()
    var = []
    for i in range(len(read_list)):
        if read_list[i][:-1]=="":
            var.append(np.array([]))
        else:
            var.append(np.array(list(map(int,read_list[i][:-1].split(" ")))))
    return var

def get_names(nodes_list, net):
    '''
    function to return the names of the nodes corresponding to a list of nodes index associated to
    a network from a VirusMultiplex object
    node_list = list of numbers associated to node IDs in the net
    net = VirusMultiplex object
    '''

    keys = np.array(list(net.node_map.keys()))
    return keys[nodes_list]