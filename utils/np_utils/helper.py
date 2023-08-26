# General Helper 
import numpy as np
import pandas as pd
from .pose_helper import pose_vec_q_to_mat

def inv(T):
    '''Take a T matrix to provide inverse(fast)
    Input: 
    T       -- Transformation matrix to take inverse on 
    Output:
    T_      -- Inverse Transformation matrix'''
    T_ = np.copy(T)
    R,t = T[:3,:3],T[:3,3:]
    T_[:3,:3] = R.T
    T_[:3,3:] = - R.T @ t
    return T_

def txt_to_4x4(fname,datas=-1):
    '''Read file of pose q arrays 
    
    Input:
    fname   -- File name to be read for q arrays
    datas   -- number of data to return (default = all)

    Output:
    data   -- np.array of all pose converted to T matrises      (nx4x4)'''
    data = np.array(pd.read_csv(fname,delimiter=' ', header = None))[:datas]
    data = np.array([pose_vec_q_to_mat(i) for i in data])
    return data

def txt_to_q(fname,datas=-1):
    '''Read file of pose q arrays 
    
    Input:
    fname   -- File name to be read for q arrays
    datas   -- number of data to return (default = all)

    Output:
    data   -- np.array of all pose converted to T matrises      (nx4x4)'''
    if datas == -1:
        data = np.array(pd.read_csv(fname,delimiter=' ', header = None))
    else:
        data = np.array(pd.read_csv(fname,delimiter=' ', header = None))[:datas]
    return data

def read_file_list(filename):
    """Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches
