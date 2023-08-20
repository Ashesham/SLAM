import pandas as pd
import math
import functools
import torch
import torch.nn.functional as F

def make_intrinsic(fx,fy,cx,cy):
    return torch.tensor([  [fx,0,cx],
                [0,fy,cy],
                [0, 0, 1]])
def T_mat(t_mat, r_mat):
    T = torch.eye(4)
    T[:3,:3] = r_mat
    T[:3,-1] = t_mat
    return T
#Basic Operations
def rot2quat(matrix: torch.Tensor) -> torch.Tensor:
    """ Taken from pytorch3d
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    x = torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    q_abs = torch.zeros_like(x)
    positive_mask = x > 0
    q_abs[positive_mask] = torch.sqrt(x[positive_mask])

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
def quat2mat(q):
    ''' Edited from 
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return torch.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return torch.tensor(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])
def euler2quat(z=0, y=0, x=0, isRadian=True):
    ''' Return quaternion corresponding to these Euler angles
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    '''
  
    if not isRadian:
        z = ((math.pi)/180.) * z
        y = ((math.pi)/180.) * y
        x = ((math.pi)/180.) * x
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return torch.tensor([
                     cx*cy*cz - sx*sy*sz,
                     cx*sy*sz + cy*cz*sx,
                     cx*cz*sy - sx*cy*sz,
                     cx*cy*sz + sx*cz*sy])
def inv(T):
    T_ = torch.clone(T)
    R,t = T[:3,:3],T[:3,3:]
    T_[:3,:3] = R.T
    T_[:3,3:] = - R.T @ t
    return T_
def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.

    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)

    Output:
    Relative 3D transformation from a to b.
    """
    return torch.dot(torch.linalg.inv(a),b) 
# Spesific for q to mat for kitti trajectory
def pose_vec_q_to_mat_kitti(vec):
    '''Convert Kitti q values (With TimeStamp) to T matrix 
    
    Input:
    vec -- poses (7)

    Output:
    Tmat -- T Matrix (4x4)'''
    tx = vec[1]
    ty = vec[2]
    tz = vec[3]
    trans = torch.tensor([tx, ty, tz]).reshape((3,1))
    q = [vec[7],vec[4], vec[5], vec[6]]
    rot = quat2mat(q)
    Tmat = torch.cat((rot, trans), axis=1)
    hfiller = torch.tensor([0, 0, 0, 1]).reshape((1,4))
    Tmat = torch.cat((Tmat, hfiller), axis=0)
    return Tmat
# convert a 4*4 mat to vec with q
def pose_mat_to_vec_q(pose_mat):
    tx = pose_mat[0, 3]
    ty = pose_mat[1, 3]
    tz = pose_mat[2, 3]
    rot = pose_mat[:3, :3]
    qw, qx, qy, qz = rot2quat(rot)

    # set time as zero for now
    time = 0
    return [time, tx, ty, tz, qx, qy, qz, qw]
def pose_vec_q_to_mat(vec):
    if len(vec)==8:
        vec=vec[1:]
    assert len(vec)==7, "Invalid dimension for pose"

    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = torch.tensor([tx, ty, tz]).reshape((3,1))
    q = [vec[6],vec[3], vec[4], vec[5]]
    rot = quat2mat(q)
    ## Edit
    # rot = rot.T
    # trans = - rot@trans
    Tmat = torch.cat((rot, trans), axis=1)
    hfiller = torch.tensor([0, 0, 0, 1]).reshape((1,4))
    Tmat = torch.cat((Tmat, hfiller), axis=0)
    return Tmat

#Eval_Utils
def compute_distance(transform):
    """    Compute the distance of the translational component of a 4x4 homogeneous matrix.

    Input:
    transform -- error44 (4x4)
    
    Output:
    trans_err -- Error in trajectory  
    """
    return torch.linalg.norm(transform[0:3,3])
def compute_angle(transform):
    """    Compute the rotation angle from a 4x4 homogeneous matrix.

    Input:
    transform -- error44 (4x4)
    
    Output:
    ang_err -- Error in orientation  
    """
    # an invitation to 3-d vision, p 27
    return torch.arccos( min(1,max(-1, (torch.trace(transform[0:3,0:3]) - 1)/2) ))
def align_to_origin(data,is_vo=False):
    """Align TartainAir or predicted pose using VO to origin 
    
    Input:
    data -- trajectory (nx4x4)
    is_vo -- Flag to specify if is VO data or TarTainAir
    
    Output:
    data -- aligned trajectory to origin (nx4x4)
    """
    viewer_origin = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    vo_origin = torch.tensor([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

    if is_vo:
        data = vo_origin@data
    else:
        data = viewer_origin@inv(data[0])@data

    return data

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
    errors = torch.tensor(errors)
    err_t = torch.tensor(err_t)
    err_r = torch.tensor(err_r)
    rmse_t = torch.sqrt(torch.dot(err_t,err_t) / len(err_t))
    rmse_r = torch.sqrt(torch.dot(err_r,err_r) / len(err_r))
    return rmse_r,rmse_t

'''def align(model,data):
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
    
    W = torch.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += torch.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = torch.linalg.linalg.svd(W.transpose())
    S = torch.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh
    trans = data.mean(1) - rot * model.mean(1)
    
    model_aligned = rot * model + trans
    alignment_error = model_aligned - data
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]

    return model_aligned,trans_error
'''
def compute_ate(gtruth_xyz, pred_xyz):    
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz[0]
    pred_xyz += offset[None,:]

    # Optimize the scaling factor
    scale = torch.sum(gtruth_xyz * pred_xyz)/torch.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = torch.sqrt(torch.sum(alignment_error ** 2))/len(pred_xyz)
    return rmse
