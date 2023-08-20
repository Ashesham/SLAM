import plotly.graph_objects as go
import numpy as np
from typing import Optional
import open3d as o3d
import time
import pandas as pd
import numpy as np
import tqdm

viewer_origin = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
vo_origin = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])


class viewer:
    def __init__(self,):
        self.prev_t = [0,0,0]
        self.prev_td = [0,0,0]
        pass#self._run()

    def update_pose(self, pose, origin, color=[1, 0, 0]):
        transformed = origin@pose
        

        if not type(self.prev_t) == list: 
            points = [self.prev_t, transformed[:3,-1]]
            lines = [[0, 1]]
            colors = [color]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(line_set)
        else:
            t1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            t1.transform(transformed)
            self.vis.add_geometry(t1)
        
        
        self.prev_t = transformed[:3,-1]

    def update_pose_data(self, pose, color=[1, 0, 0]):
        if pose.shape[-1] == 4:
            pose = pose[:3,-1]

        if not type(self.prev_td) == list: 
            points = [self.prev_td, pose]
            lines = [[0, 1]]
            colors = [color]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(line_set)
        else:
            T = np.eye(4)
            T [:3,-1] = pose
            t1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            t1.transform(T)
            self.vis.add_geometry(t1)
        
        
        self.prev_td = pose

    def update_pose_data_live(self, pose, color=[1, 0, 0]):
        if pose.shape[-1] == 4:
            pose = pose[:3,-1]

        if not type(self.prev_td) == list: 
            points = [self.prev_td, pose]
            lines = [[0, 1]]
            colors = [color]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(line_set)
        else:
            T = np.eye(4)
            T [:3,-1] = pose
            t1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            t1.transform(T)
            self.vis.add_geometry(t1)
        
        
        self.prev_td = pose
        self.update_view()

    def update_view(self):
        # cam = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        # self.vis.reset_view_point(True)
        # self.vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        self.vis.poll_events()
        self.vis.update_renderer()

    def draw_dots(self, poses, color=[1, 0, 0]):
        ''' Draw spheres in the given coordinates
        input:
        pose -- q vector or 3d location [7 or 3]

        '''
        if poses.shape[-1] == 4:
            poses = poses[:3,-1]

        for i in tqdm.tqdm(poses):
            point = o3d.geometry.TriangleMesh.create_sphere(0.05,resolution=2)
            T = np.eye(4)
            T[:3,3] = i[:3]
            point.transform(T)
            self.vis.add_geometry(point)

    def draw_dots_live(self, poses, color=[1, 0, 0]):
        ''' Draw spheres in the given coordinates
        input:
        pose -- q vector or 3d location [7 or 3]

        '''
        if poses.shape[-1] == 4:
            poses = poses[:3,-1]

        for i in tqdm.tqdm(poses):
            point = o3d.geometry.TriangleMesh.create_sphere(0.05,resolution=2)
            T = np.eye(4)
            T[:3,3] = i[:3]
            point.transform(T)
            self.vis.add_geometry(point)
            self.update_view()
            time.sleep(0.01)


    def init(self): 
        # Create Open3d visualization window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.add_geometry(coordinate_frame)

    def run(self):
        self.vis.run()
        
    def _run(self):        
        # Create Open3d visualization window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        # create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.vis.add_geometry(coordinate_frame)
        while 1:      
            self.update_view()
            time.sleep(0.01)


def init_figure(height: int = 800) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            eye=dict(x=0., y=-.1, z=-2),
            up=dict(x=0, y=-1., z=0),
            projection=dict(type="orthographic")),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode='data',
            dragmode='orbit',
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.1
        ),
    )
    return fig
    
def to_homogeneous(points):
    pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)

def plot_camera(
        fig: go.Figure,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray,
        color: str = 'rgb(0, 0, 255)',
        name: Optional[str] = None,
        legendgroup: Optional[str] = None,
        fill: bool = False,
        size: float = 1.0,
        text: Optional[str] = None):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2]*2, K[1, 2]*2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        pyramid = go.Mesh3d(
            x=x, y=y, z=z, color=color, i=i, j=j, k=k,
            legendgroup=legendgroup, name=name, showlegend=False,
            hovertemplate=text.replace('\n', '<br>'))
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([
        vertices[i] for i in triangles.reshape(-1)
    ])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x, y=y, z=z, mode='lines', legendgroup=legendgroup,
        name=name, line=dict(color=color, width=1), showlegend=False,
        hovertemplate=text.replace('\n', '<br>'))
    fig.add_trace(pyramid)

def quat2mat(q):
    ''' Calculate rotation matrix corresponding to quaternion
    https://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/quaternions.py
    Parameters
    ----------
    q : 4 element array-like

    Returns
    -------
    M : (3,3) array
      Rotation matrix corresponding to input quaternion *q*

    Notes
    -----
    Rotation matrix applies to column vectors, and is applied to the
    left of coordinate vectors.  The algorithm here allows non-unit
    quaternions.

    References
    ----------
    Algorithm from
    http://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> M = quat2mat([1, 0, 0, 0]) # Identity quaternion
    >>> np.allclose(M, np.eye(3))
    True
    >>> M = quat2mat([0, 1, 0, 0]) # 180 degree rotn around axis 0
    >>> np.allclose(M, np.diag([1, -1, -1]))
    True
    '''
    w, x, y, z = q
    Nq = w*w + x*x + y*y + z*z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array(
           [[ 1.0-(yY+zZ), xY-wZ, xZ+wY ],
            [ xY+wZ, 1.0-(xX+zZ), yZ-wX ],
            [ xZ-wY, yZ+wX, 1.0-(xX+yY) ]])

def pose_vec_q_to_mat(vec):
    if len(vec)==8:
        vec=vec[1:]
    assert len(vec)==7, "Invalid dimension for pose"

    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    q = [vec[6],vec[3], vec[4], vec[5]]
    rot = quat2mat(q)
    ## Edit
    # rot = rot.T
    # trans = - rot@trans
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat
def pose_vec_q_to_mat1(vec):
    tx = vec[0]
    ty = vec[1]
    tz = vec[2]
    trans = np.array([tx, ty, tz]).reshape((3,1))
    q = [vec[6],vec[3], vec[4], vec[5]]
    rot = quat2mat(q)
    
    Tmat = np.concatenate((rot, trans), axis=1)
    hfiller = np.array([0, 0, 0, 1]).reshape((1,4))
    Tmat = np.concatenate((Tmat, hfiller), axis=0)
    return Tmat
def inv(T):
    T_ = np.copy(T)
    R,t = T[:3,:3],T[:3,3:]
    T_[:3,:3] = R.T
    T_[:3,3:] = - R.T @ t
    return T_



def display_poses_dataset(window,f_name='./data/pose_left.txt',color=[1,0,0],is_vo=False,data_origin=False):
    global viewer_origin, vo_origin
    dcsv = np.array(pd.read_csv(f_name,delimiter=' '))
    Tdcsv = np.array([pose_vec_q_to_mat(i) for i in dcsv])
    if is_vo:
        origin = vo_origin
    elif data_origin:
        origin = viewer_origin@inv(Tdcsv[0])
    else:
        origin = viewer_origin

    for i in range(len(Tdcsv)):
        window.update_pose(Tdcsv[i], origin,color)
    window.prev_t = [0,0,0]

def display_poses(window,data,color=[1,0,0]):
    for i in range(len(data)):
        window.update_pose_data(data[i], color)
    window.prev_td = [0,0,0]

def correct_scale(gtfname,predfname,datas=-1):
    global viewer_origin, vo_origin

    gt = np.array(pd.read_csv(gtfname,delimiter=' '))[:datas]
    gt = np.array([pose_vec_q_to_mat(i) for i in gt])
    pred = np.array(pd.read_csv(predfname,delimiter=' '))[:datas]
    pred = np.array([pose_vec_q_to_mat(i) for i in pred])

    gt = viewer_origin@inv(gt[0])@gt
    pred = vo_origin@pred

    gt, pred = gt[:,:3,-1],pred[:,:3,-1]

    offset = gt[0] - pred[0]

    pred += offset[None,:]

    # Optimize the scaling factor
    scale = np.sum(gt * pred)/np.sum(pred ** 2)
    pred = pred*scale
    alignment_error = pred - gt
    #average translation error
    rmse = np.sqrt(np.sum(alignment_error ** 2))/len(gt)
    return pred,rmse
