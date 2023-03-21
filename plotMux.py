import numpy as np
import graph_tool as gt
import matplotlib.pyplot as plt

def plotMultiplex(g_list, g_agg, positions = None, elev=20, azim=10, label=""):
    nodes_list = []
    centr_list = []
    if positions is None:
        positions = gt.draw.sfdp_layout(g_agg)
        positions = positions.get_2d_array([0,1])
    
    for i in range(len(g_list)):
        centr_list.append(g_list[i].get_total_degrees(g_list[i].get_vertices()))
        nodes_list.append(g_list[i].get_vertices()[centr_list[i][centr_list[i]!=0]])
        

    ax = plt.figure(figsize=(12,12)).add_subplot(projection='3d')
    xx, yy = np.meshgrid(np.linspace(positions[0].min(), positions[0].max(),2), np.linspace(positions[1].min(), positions[1].max(),2))
    X =  xx
    Z =  yy
    for i in range(len(g_list)):
        sizes = centr_list[i]
        sizes = sizes-min(sizes)
        sizes =sizes/max(sizes)*50
        #sizes[sizes<5]=0.1
        ind = np.argsort(sizes)[-100:]
        ax.scatter(positions[0][ind], positions[1][ind], zs=i, zdir='y', label=label,s=sizes[ind])
        Y =  i*np.ones(X.shape)
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1, alpha=0.5)

    # Make legend, set axes limits and labels
    ax.set_xlim(positions[0].min(), positions[0].max())
    ax.set_ylim(0, len(g_list))
    ax.set_zlim(positions[1].min(), positions[1].max())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis("off")
    #ax.legend(loc=(0.3,0.87))

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=elev, azim=azim)

    plt.show()

    return positions