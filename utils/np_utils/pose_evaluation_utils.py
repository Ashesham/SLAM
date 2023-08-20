# Some of the code are from the TUM evaluation toolkit:
# https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#absolute_trajectory_error_ate

import math
import numpy as np
import functools
import pandas as pd
from .pose_helper import *
from .helper import associate

def compute_distance(transform):
    """    Compute the distance of the translational component of a 4x4 homogeneous matrix.

    Input:
    transform -- error44 (4x4)
    
    Output:
    trans_err -- Error in trajectory  
    """
    return np.linalg.norm(transform[0:3,3])

def compute_angle(transform):
    """    Compute the rotation angle from a 4x4 homogeneous matrix.

    Input:
    transform -- error44 (4x4)
    
    Output:
    ang_err -- Error in orientation  
    """
    # an invitation to 3-d vision, p 27
    return np.arccos( min(1,max(-1, (np.trace(transform[0:3,0:3]) - 1)/2) ))

def find_rpe(gt,pred):
    """Find rpe of a single pair of poses 
    
    Input:
    gt -- gt pose (4x4)
    pred -- predicted trajectory (4x4)
    
    Output:
    error44 -- elementwise rpe error
    rmse_r -- rmse rpe rotational error of the trajectory 
    rmse_t -- rmse rpe translational error of the trajectory 
    """
    error44 = ominus(ominus( pred[0], pred[1] ),
                            ominus( gt[0], gt[1] ) )
    trans = compute_distance(error44)
    rot = compute_angle(error44)    

    return error44,rot,trans

def find_all_rpe(gts,preds):
    """Find rmse rpe of a continous trajectory of poses 
    
    Input:
    gts -- gt trajectory (nx4x4)
    preds -- predicted trajectory [aligned] (nx4x4)
    
    Output:
    rmse_r -- rmse rpe rotational error of the trajectory 
    rmse_t -- rmse rpe translational error of the trajectory 
    """
    assert len(gts) == len(preds), "The data from preds and gts doesn't match"
    errors, err_t, err_r = [], [], []
    for i in range(1,len(gts)):
        t1,t2,t3 = find_rpe([gts[i-1],gts[i]],[preds[i-1],preds[i]])
        errors.append(t1)
        err_t.append(t2)
        err_r.append(t3)
    errors = np.array(errors)
    err_t = np.array(err_t)
    err_r = np.array(err_r)
    rmse_t = np.sqrt(np.dot(err_t,err_t) / len(err_t))
    rmse_r = np.sqrt(np.dot(err_r,err_r) / len(err_r))
    return rmse_r,rmse_t

def align_to_origin(data,is_vo=False):
    """Align TartainAir or predicted pose using VO to origin 
    
    Input:
    data -- trajectory (nx4x4)
    is_vo -- Flag to specify if is VO data or TarTainAir
    
    Output:
    data -- aligned trajectory to origin (nx4x4)
    """
    viewer_origin = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    vo_origin = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

    if is_vo:
        data = vo_origin@data
    else:
        data = viewer_origin@inv(data[0])@data

    return data

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn) [Trajectory]
    data -- second trajectory (3xn)
    
    Output:
    model_aligned -- model aligned both in orientation and scale (3xn)
    trans_error -- translational error [rmse]
    
    """
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    
    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return model_aligned,trans_error

def compute_ate(gtruth_file, pred_file):
    gtruth_list = read_file_list(gtruth_file)
    pred_list = read_file_list(pred_file)
    matches = associate(gtruth_list, pred_list, 0, 0.01)
    if len(matches) < 2:
        return False

    gtruth_xyz = np.array([[float(value) for value in gtruth_list[a][0:3]] for a,b in matches])
    pred_xyz = np.array([[float(value) for value in pred_list[b][0:3]] for a,b in matches])
    
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz[0]
    pred_xyz += offset[None,:]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz)/np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2))/len(matches)
    return rmse

