import csv, binascii
import numpy as np
import math
import os
from evo.core.trajectory import PoseTrajectory3D
from evo.core import sync
import typing
import torch

torch.set_printoptions(precision=6)

def csv_read_matrix(file_path, delim=',', comment_str="#"):
    """
    directly parse a csv-like file into a matrix
    :param file_path: path of csv file (or file handle)
    :param delim: delimiter character
    :param comment_str: string indicating a comment line to ignore
    :return: 2D list with raw data (string)
    """
    if hasattr(file_path, 'read'):  # if file handle
        generator = (line for line in file_path
                     if not line.startswith(comment_str))
        reader = csv.reader(generator, delimiter=delim)
        mat = [row for row in reader]
    else:
        skip_3_bytes = has_utf8_bom(file_path)
        with open(file_path) as f:
            if skip_3_bytes:
                f.seek(3)
            generator = (line for line in f
                         if not line.startswith(comment_str))
            reader = csv.reader(generator, delimiter=delim)
            mat = [row for row in reader]
    return mat

def has_utf8_bom(file_path):
    """
    Checks if the given file starts with a UTF8 BOM
    wikipedia.org/wiki/Byte_order_mark
    """
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 3:
        return False
    with open(file_path, 'rb') as f:
        return not int(binascii.hexlify(f.read(3)), 16) ^ 0xEFBBBF

def get_trajs(f_path):
    raw_mat = csv_read_matrix(f_path,delim=' ')
    error_msg = "TUM trajectory files must have 8 entries per row and no trailing delimiter at the end of the rows (space)"
    if not raw_mat or (len(raw_mat) > 0 and len(raw_mat[0]) != 8):
        raise error_msg
    try:
        mat = np.array(raw_mat).astype(float)
    except ValueError:
        raise error_msg
    stamps = mat[:, 0]  # n x 1
    xyz = mat[:, 1:4]  # n x 3
    quat = mat[:, 4:]  # n x 4
    quat = np.roll(quat, 1, axis=1)  # shift 1 column -> w in front column
    return PoseTrajectory3D(xyz,quat,stamps)

def sim3(r: torch.tensor, t: torch.tensor, s: float) -> torch.tensor:
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :param s: positive, non-zero scale factor
    :return: Sim(3) similarity transformation matrix
    """
    sim3 = torch.eye(4)
    sim3[:3, :3] = s * r
    sim3[:3, 3] = t
    return sim3

def relative_se3(p1: torch.tensor, p2: torch.tensor) -> torch.tensor:
    """
    :param p1, p2: SE(3) matrices
    :return: the relative transformation p1^{⁻1} * p2
    """
    return se3_inverse(p1)@ p2

def se3_inverse(p: torch.tensor) -> torch.tensor:
    """
    :param p: absolute SE(3) pose
    :return: the inverted pose
    """
    r_inv = p[:3, :3].T
    t_inv = -r_inv@p[:3, 3]
    return se3(r_inv, t_inv)

def se3(r: torch.tensor = torch.eye(3),
        t: torch.tensor = torch.tensor([0, 0, 0])) -> torch.tensor:
    """
    :param r: SO(3) rotation matrix
    :param t: 3x1 translation vector
    :return: SE(3) transformation matrix
    """
    se3 = torch.eye(4,dtype=torch.float64)
    se3[:3, :3] = r
    se3[:3, 3] = t
    return se3

def umeyama_alignment(x: torch.tensor, y: torch.tensor,
                      with_scale: bool = False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise "data matrices must have the same shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (torch.linalg.norm(x - mean_x[:, None])**2)

    # covariance matrix, eq. 38
    outer_sum = torch.zeros((m, m))
    for i in range(n):
        outer_sum += torch.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = 1.0 / n * outer_sum

    # SVD (text betw. eq. 38 and 39)
    u, d, v = torch.linalg.svd(cov_xy)
    if torch.count_nonzero(d > torch.finfo(d.dtype).eps) < m - 1:
        raise "Degenerate covariance rank, Umeyama alignment is not possible"

    # S matrix, eq. 43
    s = torch.eye(m)
    if torch.linalg.det(u) * torch.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = ((u@s)@v).to(torch.float64)
    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * torch.trace(torch.diag(d)@s) if with_scale else 1.0
    t = mean_y - torch.multiply(c, r@mean_x)

    return r, t, c

def scale(s: float, _poses_se3, _positions_xyz) -> None:
    """
    apply a scaling to the whole path
    :param s: scale factor
    """
    # if hasattr(self, "_poses_se3"):
    _poses_se3 = [
        se3(p[:3, :3], s * p[:3, 3]) for p in _poses_se3
    ]
    # if hasattr(self, "_positions_xyz"):
    _positions_xyz = s * _positions_xyz

    return _poses_se3, _positions_xyz

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = torch.array(matrix, dtype=torch.float64, copy=False)[:4, :4]
    if isprecise:
        q = torch.empty((4, ))
        t = torch.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = torch.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = torch.linalg.eigh(K)
        q = V[[3, 0, 1, 2], torch.argmax(w)]
    if q[0] < 0.0:
        torch.negative(q, q)
    return q

def se3_poses_to_xyz_quat_wxyz(
    poses: typing.Sequence[torch.tensor]
) -> typing.Tuple[torch.tensor, torch.tensor]:
    xyz = torch.tensor([pose[:3, 3] for pose in poses],requires_grad=True)
    quat_wxyz = torch.tensor([quaternion_from_matrix(pose) for pose in poses],requires_grad=True)
    return xyz, quat_wxyz

def transform(t: torch.tensor, poses_se3, right_mul: bool = False,
                propagate: bool = False) -> torch.tensor:
    """
    apply a left or right multiplicative transformation to the whole path
    :param t: a 4x4 transformation matrix (e.g. SE(3) or Sim(3))
    :param right_mul: whether to apply it right-multiplicative or not
    :param propagate: whether to propagate drift with RHS transformations
    """
    num_poses = len(poses_se3)
    if right_mul and not propagate:
        # Transform each pose individually.
        _poses_se3 = [p@t for p in poses_se3]
    elif right_mul and propagate:
        # Transform each pose and propagate resulting drift to the next.
        ids = torch.arange(0, num_poses, 1)
        rel_poses = [
            relative_se3(poses_se3[i], poses_se3[j]).dot(t)
            for i, j in zip(ids, ids[1:])
        ]
        _poses_se3 = [poses_se3[0]]
        for i, j in zip(ids[:-1], ids):
            _poses_se3.append(_poses_se3[j].dot(rel_poses[i]))
    else:
        _poses_se3 = [t@p for p in poses_se3]
    return _poses_se3

def align_traj(traj_ref,traj_est, correct_scale: bool = False,
            correct_only_scale: bool = False, n: int = -1):
    """
    align to a reference trajectory using Umeyama alignment
    :param traj_ref: reference trajectory
    :param correct_scale: set to True to adjust also the scale
    :param correct_only_scale: set to True to correct the scale, but not the pose
    :param n: the number of poses to use, counted from the start (default: all)
    :return: aligned trajectory
    """
    with_scale = correct_scale or correct_only_scale
    if n == -1:
        r_a, t_a, s = umeyama_alignment(traj_est[:,:3,3].T, traj_ref[:,:3,3].T, with_scale)
    else:
        r_a, t_a, s = umeyama_alignment(
            traj_est[:n,:3,3].T, traj_est[:n, :3,3].T,
            with_scale)

    poses_se3, positions_xyz = traj_est, traj_est[:,:3,3]

    if correct_only_scale:
        poses_se3, positions_xyz = scale(s,poses_se3,positions_xyz)
    elif correct_scale:
        poses_se3, positions_xyz = scale(s,poses_se3,positions_xyz)
        poses_se3 = transform(se3(r_a, t_a),poses_se3)
    else:
        poses_se3 = transform(se3(r_a, t_a),poses_se3)
    # traj_est.poses_se3, traj_est.positions_xyz = poses_se3, positions_xyz 

    return poses_se3

def align_origin_traj(traj_ref, traj_est) -> torch.tensor:
        """
        align the origin to the origin of a reference trajectory
        :param traj_ref: reference trajectory
        :return: the used transformation
        """
        traj_origin = traj_est[0]
        traj_ref_origin = traj_ref[0]
        to_ref_origin = traj_ref_origin@se3_inverse(traj_origin)
        return to_ref_origin

def ape(traj_ref, traj_est, align: bool = False, correct_scale: bool = False, 
        n_to_align: int = -1, align_origin: bool = False):

    # Align the trajectories.
    only_scale = correct_scale and not align
    
    if align or correct_scale:
        traj_est = align_traj(traj_ref,traj_est, correct_scale, only_scale, n=n_to_align)
    elif align_origin:
        alignment_transformation = align_origin_traj(traj_ref, traj_est)
        traj_est = transform(alignment_transformation,traj_est)
    # Calculate APE.
    data = (traj_ref, traj_est)
    E = [
        relative_se3(x_t, x_t_star) for x_t, x_t_star in zip(
            traj_est, traj_ref)
    ]
    
    error = torch.tensor(
        [(E_i - torch.eye(4)).pow(2).sum().sqrt() for E_i in E], requires_grad=True)

    squared_errors = torch.pow(error, 2)
    rmse = torch.sqrt(torch.mean(squared_errors))
    
    return rmse

if __name__=='__main__':
    groundtruth_file = '/home/rp2/Projects/slambook2/ch4/example/groundtruth.txt'
    estimated_file = '/home/rp2/Projects/slambook2/ch4/example/estimated.txt'

    traj_ref, traj_est, ref_name, est_name = get_trajs(groundtruth_file), get_trajs(estimated_file), groundtruth_file, estimated_file
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    traj_ref, traj_est = torch.tensor(traj_ref.poses_se3, requires_grad=True, dtype=torch.float64), torch.tensor(traj_est.poses_se3, requires_grad=True, dtype=torch.float64)
    res = ape(traj_ref=traj_ref, traj_est=traj_est,align=True, correct_scale=True)
    print(res)