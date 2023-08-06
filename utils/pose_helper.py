import numpy as np
import pandas as pd

def make_intrinsic(fx,fy,cx,cy):
    return np.array([  [fx,0,cx],
                [0,fy,cy],
                [0, 0, 1]])

class Traj_helper:
    def __init__(self):
        pass

    def read_traj(self,fname, delimited=' ', have_TS=False):
        data = pd.read_csv('./Tartal_backup/P000/pose_left.txt',delimiter=' ')
        if have_TS:
            pass

def T_mat(t_mat, r_mat):
    T = np.eye(4)
    T[:3,:3] = r_mat
    T[:3,-1] = t_mat
    return T

