import numpy as np

from utils.pose_helper import make_intrinsic, Traj_helper
from utils.viz_helper import *

import warnings
warnings.filterwarnings("ignore")


if __name__=='__main__':
    helper = Traj_helper()
    window = viewer()
    window.run()

    display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory_night/P001.txt',color=[0,0.5,0],data_origin=True)
    # display_poses_dataset(window,f_name='./data/traj_Factory_night_0001.txt',color=[1,0,0],is_vo=True)

    scale_corrected, err = correct_scale('./data/Trajs/abandonedfactory_night/P001.txt','./data/traj_Factory_night_0001.txt',datas=-1)
    print("Error: ",err)
    display_poses(window,scale_corrected,color=[1,0,0])


    '''# display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory/P000.txt',color=[0.5,0,0])
    # display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory_night/P000.txt',color=[1,0,0])

    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest/P001.txt',color=[0,0.5,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest/P002.txt',color=[0,0.5,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest/P004.txt',color=[0,0.5,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest/P005.txt',color=[0,0.5,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest/P006.txt',color=[0,0.5,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P010.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P011.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P012.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P013.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P014.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P015.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P016.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P017.txt',color=[0,1,0])
    # display_poses_dataset(window,f_name='./data/Trajs/seasonsforest_winter/P018.txt',color=[0,1,0])

    # display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory/P010.txt',color=[0,0,0.5])
    # display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory_night/P010.txt',color=[0,0,1])
    # display_poses_dataset(window,f_name='./data/Trajs/abandonedfactory/P002.txt',color=[0,0,1])

    # display_poses_dataset(window,f_name='./data/traj.txt',color=[0,0,1],is_vo=True)'''

    window.vis.run()