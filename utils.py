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