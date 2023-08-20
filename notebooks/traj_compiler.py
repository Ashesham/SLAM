
import glob
import sys
sys.path.append('..')
from utils.np_utils.helper import txt_to_q
import tqdm
import numpy as np
from utils.np_utils.pose_evaluation_utils import *
from utils.np_utils.pose_helper import *
from utils.np_utils.helper import *
import pandas as pd

from utils.np_utils.pose_helper import make_intrinsic, Traj_helper
from utils.np_utils.viz_helper import *


All_data = {}
for env in sorted(glob.glob('../data/Trajs/*')):
    temp = []
    for fnames in sorted(glob.glob(env+'/P*.txt')):
        if len(temp)==0:
            temp=txt_to_q(fnames)
        else:
            temp = np.concatenate((temp,txt_to_q(fnames)))
    All_data[env.split('/')[-1]] = np.array(temp)
    

# https://stackoverflow.com/questions/37742358/sorting-points-to-form-a-continuous-line
import matplotlib.pyplot as plt
def order_points(points, ind):
    idxs = [i for i in range(len(points))]
    updated_id = [idxs.pop(ind)]
    points_new = [ points.pop(ind) ]  # initialize a new list of points with the known first point
    pcurr      = points_new[-1]       # initialize the current point (as the known point)

    i_xyz = np.array([3,7,11,15])
    while len(points)>0:
        d      = np.linalg.norm(np.array(points)[:,i_xyz] - np.array(pcurr)[i_xyz], axis=1)  # distances between pcurr and all other remaining points
        ind    = d.argmin()                   # index of the closest point
        
        if abs(d[ind])>2:
            return points_new, updated_id
            
        d      = np.linalg.norm(np.array(points) - np.array(pcurr), axis=1)  # distances between pcurr and all other remaining points
        ind    = d.argmin()                   # index of the closest point
        points_new.append( points.pop(ind) )  # append the closest point to points_new
        updated_id.append(idxs.pop(ind))
        pcurr  = points_new[-1]               # update the current point
    return points_new, updated_id
# create sine curve:
# x      = np.linspace(0, 2 * np.pi, 100)
# y      = np.sin(x)
# xs = data[:,0]#np.linspace(0, 2 * np.pi, 1000)
# ys = data[:,1]#np.sin(x)
# zs = data[:,2]

data = np.concatenate((All_data['abandonedfactory'],All_data['abandonedfactory_night']))
# x,y,z,q1,q2,q3,q4 = data.T

# # shuffle the order of the x and y coordinates:
# idx    = np.random.permutation(x.size)
# xs,ys,zs  = x[idx], y[idx],   # shuffled points

d1 = np.copy(data)
idx = np.random.permutation(len(data))
data = [pose_vec_q_to_mat(i) for i in data[idx]]

# find the least point:
maax = -1000000
for i in range(len(data)):
    if np.linalg.norm(data[i])>maax:
        ind = i
        maax = np.linalg.norm(data[i])
#data.argmin()

# assemble the x and y coordinates into a list of (x,y) tuples:
points = list(data)#[(xx,yy,zz)  for xx,yy,zz in zip(x,y,z)]
# points = list(np.array(points).reshape((len(points),16)))
mat = []
for i in tqdm.tqdm(points):
    mat.append([])
    for j in points:
        mat[-1].append(np.linalg.norm(inv(i)@(j)-np.eye(4)))

import numpy as np
import networkx as nx
import itertools

# Replace this with your actual distance matrix
distances = np.array(mat)

# Create a complete graph using the distances matrix
G = nx.Graph()
n = distances.shape[0]  # Number of nodes

for i in range(n):
    for j in range(i+1, n):
        G.add_edge(i, j, weight=distances[i][j])

# Use the asadpour_atsp algorithm from the networkx's approximation module
shortest_tour = nx.approximation.christofides(G)


# # Find the optimal TSP tour using the Held-Karp algorithm
# shortest_tour = nx.algorithms.approximation.asadpour_atsp(G, source=0)

# Calculate the total distance of the tour
total_distance = sum(distances[shortest_tour[i]][shortest_tour[i+1]] for i in range(len(shortest_tour) - 1))

# print("Optimal TSP tour:", shortest_tour)
print("Total distance:", total_distance)


# order the points based on the known first point:
iid = shortest_tour
points_new  = np.array(points)[iid]

# plot:
fig,ax = plt.subplots(1, 2, figsize=(10,4))
data  = np.array(points_new)
ax[0].plot(d1[:,0], d1[:,1])  # original (shuffled) points
ax[1].plot(data[:,0,3], data[:,1,3])  # new (ordered) points
# ax[2].plot([i for i in range(500)], iid[:500])  # original (shuffled) points
ax[0].set_title('Original')
ax[1].set_title('Ordered')
plt.tight_layout()
plt.show()
